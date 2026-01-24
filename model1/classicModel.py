import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class vitModel(nn.Module):
    def __init__(self, sequenceLength: int, inFeatures: int=11, pretrained=True):
        super(vitModel, self).__init__()
        
        self.sequenceLength = sequenceLength
        self.inFeatures = inFeatures

        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)

        self.vit.conv_proj = nn.Conv2d(in_channels=1, out_channels=768, kernel_size=(1, inFeatures), stride=(1, 1), bias=False)
        num_tokens = sequenceLength + 1
        self.vit.encoder.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, 768))
        self.vit.heads = nn.Identity()

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
    def __init__(self, inputSize, hiddenSize, numLayers):
        super(lstmModel, self).__init__()
        self.lstm = nn.LSTM(inputSize, hiddenSize, numLayers, batch_first=True)

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        return out

class biLSTMModel(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers):
        super(biLSTMModel, self).__init__()
        self.lstm = nn.LSTM(inputSize, hiddenSize, numLayers, batch_first=True, bidirectional=True)

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
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
    def __init__(self, sequenceLength: int, inFeatures: int=11):
        super(TrafficModel, self).__init__()
        
        self.embb = sliceBlock(
            visualModel=vitModel(sequenceLength=sequenceLength, inFeatures=inFeatures),
            sequenceModel=lstmModel(inputSize=768, hiddenSize=128, numLayers=1)
        )
        self.mmtc = sliceBlock(
            visualModel=vitModel(sequenceLength=sequenceLength, inFeatures=inFeatures),
            sequenceModel=biLSTMModel(inputSize=768, hiddenSize=64, numLayers=1)
        )
        self.urllc = sliceBlock(
            visualModel=vitModel(sequenceLength=sequenceLength, inFeatures=inFeatures),
            sequenceModel=biLSTMModel(inputSize=768, hiddenSize=64, numLayers=1)
        )

        self.head_embb = nn.Linear(128, 1)
        self.head_mmtc = nn.Linear(128, 1)
        self.head_urllc = nn.Linear(128, 1)

        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(32, 1))
    
    def forward(self, x_embb, x_mmtc, x_urllc):
        feat_embb = self.embb(x_embb)
        feat_mmtc = self.mmtc(x_mmtc)
        feat_urllc = self.urllc(x_urllc)
        
        combinedFeatures = torch.cat([feat_embb, feat_mmtc, feat_urllc], dim=1)
        attnOutput, _ = self.attention(combinedFeatures, combinedFeatures, combinedFeatures)

        len_embb = feat_embb.size(1)
        len_mmtc = feat_mmtc.size(1)
        '''
        pred_embb = self.head_embb(feat_embb[:, -1, :] + attnOutput[:, len_embb - 1, :])
        pred_mmtc = self.head_mmtc(feat_mmtc[:, -1, :] + attnOutput[:, len_embb + len_mmtc - 1, :])
        pred_urllc = self.head_urllc(feat_urllc[:, -1, :] + attnOutput[:, -1, :])
        '''
        pred_embb = self.head_embb(attnOutput[:, len_embb - 1, :])
        pred_mmtc = self.head_mmtc(attnOutput[:, len_embb + len_mmtc - 1, :])
        pred_urllc = self.head_urllc(attnOutput[:, -1, :])

        attnOutput_pooled = attnOutput.mean(dim=1)  # Aggregate over sequence dimension
        ffOutput = self.ff(attnOutput_pooled)
        pred_global = self.fc(ffOutput)
              
        
        
        
        return pred_global, pred_embb, pred_mmtc, pred_urllc