import torch
import torch.nn as nn
import math


class TinyViT(nn.Module):
    """
    Custom lightweight ViT for time-series input (seq_len, inFeatures).
    Replaces torchvision vit_b_16 (86M params) with a ~1-2M param transformer.

    Architecture:
      - Linear projection: (inFeatures) -> (d_model)
      - Learnable CLS token + positional embedding
      - nn.TransformerEncoder (n_layers layers)
      - Output: (batch, seq_len+1, d_model)
    """

    def __init__(self, sequenceLength: int, inFeatures: int = 11,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model

        # Linear projection replaces conv_proj — no need to pretend input is an image
        self.input_proj = nn.Linear(inFeatures, d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Learnable positional embedding for (CLS + seq_len) tokens
        self.pos_embedding = nn.Parameter(torch.randn(1, sequenceLength + 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer norm on output (standard ViT practice)
        self.ln = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        # Xavier uniform for linear projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        # Truncated normal for positional embedding and CLS token
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (batch, seq_len, inFeatures)
        batch_size = x.shape[0]

        # Project input features to d_model
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch, seq_len+1, d_model)

        # Add positional embedding
        x = x + self.pos_embedding

        # Transformer encoder
        x = self.encoder(x)  # (batch, seq_len+1, d_model)

        x = self.ln(x)

        return x


class lstmModel(nn.Module):
    def __init__(self, inputSize, hiddenSize1=128, hiddenSize2=64, dropout=0.2):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.visualModel = visualModel
        self.sequenceModel = sequenceModel

    def forward(self, x):
        visualFeatures = self.visualModel(x)
        sequenceFeatures = self.sequenceModel(visualFeatures)
        return sequenceFeatures


class TrafficModel(nn.Module):
    def __init__(self, sequenceLength: int, inFeatures: int = 11, sliceType='embb',
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
                 dim_feedforward: int = 512, vit_dropout: float = 0.1):
        super().__init__()

        if sliceType == 'embb':
            self.backbone = sliceBlock(
                visualModel=TinyViT(
                    sequenceLength=sequenceLength, inFeatures=inFeatures,
                    d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                    dim_feedforward=dim_feedforward, dropout=vit_dropout,
                ),
                sequenceModel=lstmModel(inputSize=d_model, hiddenSize1=128, hiddenSize2=64),
            )
            out_dim = 64
        elif sliceType == 'mmtc':
            self.backbone = sliceBlock(
                visualModel=TinyViT(
                    sequenceLength=sequenceLength, inFeatures=inFeatures,
                    d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                    dim_feedforward=dim_feedforward, dropout=vit_dropout,
                ),
                sequenceModel=biLSTMModel(inputSize=d_model, hiddenSize1=128, hiddenSize2=64),
            )
            out_dim = 128  # bidirectional: 64 * 2
        elif sliceType == 'urllc':
            self.backbone = sliceBlock(
                visualModel=TinyViT(
                    sequenceLength=sequenceLength, inFeatures=inFeatures,
                    d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                    dim_feedforward=dim_feedforward, dropout=vit_dropout,
                ),
                sequenceModel=biLSTMModel(inputSize=d_model, hiddenSize1=128, hiddenSize2=64),
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
            nn.Linear(32, 1),
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
