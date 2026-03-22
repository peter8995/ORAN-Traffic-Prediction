import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class vitModel(nn.Module):
    def __init__(self, sequenceLength: int, inFeatures: int=11, pretrained=False, freeze=False):
        super(vitModel, self).__init__()

        self.sequenceLength = sequenceLength
        self.inFeatures = inFeatures

        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)

        self.vit.conv_proj = nn.Conv2d(in_channels=1, out_channels=768, kernel_size=(1, inFeatures), stride=(1, 1), bias=False)
        num_tokens = sequenceLength + 1
        self.vit.encoder.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, 768))
        self.vit.heads = nn.Identity()

        if freeze:
            # Freeze all ViT encoder layers except the last 2 transformer blocks
            for param in self.vit.parameters():
                param.requires_grad = False
            # Unfreeze conv_proj (adapted for our input shape)
            for param in self.vit.conv_proj.parameters():
                param.requires_grad = True
            # Unfreeze positional embedding (adapted for our sequence length)
            self.vit.encoder.pos_embedding.requires_grad = True
            # Unfreeze last 2 encoder blocks for fine-tuning
            for block in self.vit.encoder.layers[-2:]:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.vit.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        batch_size = x.shape[0]
        cls_tokens = self.vit.class_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        #x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder(x)
        x = self.vit.heads(x)
        return x

class lstmModel(nn.Module):
    def __init__(self, inputSize, hiddenSize1=128, hiddenSize2=64, dropout=0.2):
        super(lstmModel, self).__init__()
        self.lstm1 = nn.LSTM(inputSize, hiddenSize1, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hiddenSize1, hiddenSize2, num_layers=1, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        return out

class biLSTMModel(nn.Module):
    def __init__(self, inputSize, hiddenSize1=128, hiddenSize2=64, dropout=0.2):
        super(biLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(inputSize, hiddenSize1, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hiddenSize1 * 2, hiddenSize2, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        return out

class sliceBlock(nn.Module):
    def __init__(self, visualModel, sequenceModel):
        super(sliceBlock, self).__init__()
        self.visualModel = visualModel
        self.sequenceModel = sequenceModel
    
    def forward(self, x):
        visualFeatures = self.visualModel(x)
        sequenceFeatures = self.sequenceModel(visualFeatures)
        return sequenceFeatures
    
class TrafficModel(nn.Module):
    def __init__(self, sequenceLength: int, inFeatures: int=11, sliceType='embb'):
        super(TrafficModel, self).__init__()
        
        if sliceType == 'embb':
            self.backbone = sliceBlock(
                visualModel=vitModel(sequenceLength=sequenceLength, inFeatures=inFeatures),
                sequenceModel=lstmModel(inputSize=768, hiddenSize1=128, hiddenSize2=64)
            )
            out_dim = 64
        elif sliceType == 'mmtc':
            self.backbone = sliceBlock(
                visualModel=vitModel(sequenceLength=sequenceLength, inFeatures=inFeatures),
                sequenceModel=biLSTMModel(inputSize=768, hiddenSize1=128, hiddenSize2=64)
            )
            out_dim = 128  # bidirectional: 64 * 2
        elif sliceType == 'urllc':
            self.backbone = sliceBlock(
                visualModel=vitModel(sequenceLength=sequenceLength, inFeatures=inFeatures),
                sequenceModel=biLSTMModel(inputSize=768, hiddenSize1=128, hiddenSize2=64)
            )
            out_dim = 128  # bidirectional: 64 * 2

        self.attention = nn.MultiheadAttention(embed_dim=out_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.ff = nn.Sequential(
            nn.Linear(out_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
        )
        self.fc = nn.Sequential(nn.Linear(32, 1))

        # Spike head for multi-task learning (Binary Classification)
        # No Sigmoid here — BCEWithLogitsLoss handles it for numerical stability
        self.spike_head = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        feat = self.backbone(x)
        attnOutput, _ = self.attention(feat, feat, feat)
        context_vector = attnOutput[:, -1, :]
        
        # Regression task (Resource demand)
        ffOutput = self.ff(context_vector)
        pred = self.fc(ffOutput)
        
        # Anomaly detection task (Spike probability)
        spike_pred = self.spike_head(context_vector)
                      
        return pred, spike_pred