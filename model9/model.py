from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBranch(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        rnn_type: str,
        lstm_hidden: int,
        lstm_layers: int,
        lstm_dropout: float,
        bilstm_input_proj_dim: int,
    ) -> None:
        super().__init__()
        if rnn_type not in {"lstm", "bilstm"}:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        bidirectional = rnn_type == "bilstm"
        rnn_dropout = lstm_dropout if lstm_layers > 1 else 0.0
        self.input_proj = nn.Linear(1, bilstm_input_proj_dim)
        self.rnn = nn.LSTM(
            input_size=bilstm_input_proj_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden * (2 if bidirectional else 1), hidden_dim),
            nn.ReLU(),
            nn.Dropout(lstm_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, num_nodes = x.shape
        node_series = x.transpose(1, 2).reshape(batch_size * num_nodes, sequence_length, 1)
        node_series = self.input_proj(node_series)
        node_repr, _ = self.rnn(node_series)
        last_step = node_repr[:, -1, :]
        projected = self.output_proj(last_step)
        return projected.reshape(batch_size, num_nodes, -1)


class GATLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
        head_merge: str = "mean",
    ) -> None:
        super().__init__()
        if head_merge not in {"mean", "concat"}:
            raise ValueError(f"Unsupported head_merge: {head_merge}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.head_merge = head_merge
        self.dropout = nn.Dropout(dropout)
        if head_merge == "concat":
            if output_dim % heads != 0:
                raise ValueError(
                    f"output_dim={output_dim} must be divisible by heads={heads} when head_merge='concat'"
                )
            self.per_head_dim = output_dim // heads
        else:
            self.per_head_dim = output_dim

        self.input_proj = nn.Linear(input_dim, heads * self.per_head_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, self.per_head_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, self.per_head_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        projected = self.input_proj(x)
        projected = projected.view(batch_size, num_nodes, self.heads, self.per_head_dim)
        projected = projected.permute(0, 2, 1, 3)

        src_logits = (projected * self.attn_src.unsqueeze(0).unsqueeze(2)).sum(dim=-1)
        dst_logits = (projected * self.attn_dst.unsqueeze(0).unsqueeze(2)).sum(dim=-1)
        scores = self.leaky_relu(src_logits.unsqueeze(-1) + dst_logits.unsqueeze(-2))

        attn_mask = adjacency.to(dtype=torch.bool, device=x.device).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~attn_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        aggregated = torch.einsum("bhij,bhjd->bhid", weights, projected)
        if self.head_merge == "concat":
            aggregated = aggregated.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, self.output_dim)
        else:
            aggregated = aggregated.mean(dim=1)
        return aggregated + self.bias


class SpatialBranch(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        hidden_dim: int,
        gat_hidden: int,
        gat_heads: int,
        gat_layers: int,
        gat_dropout: float,
        gat_input_proj_dim: int,
        gat_head_merge: str,
        gat_final_head_merge: str,
    ) -> None:
        super().__init__()
        if gat_layers < 1:
            raise ValueError("gat_layers must be >= 1")

        self.input_proj = nn.Linear(sequence_length, gat_input_proj_dim)
        self.input_dropout = nn.Dropout(gat_dropout)
        self.layers = nn.ModuleList()

        for layer_idx in range(gat_layers):
            in_dim = gat_input_proj_dim if layer_idx == 0 else gat_hidden
            out_dim = hidden_dim if layer_idx == gat_layers - 1 else gat_hidden
            head_merge = gat_final_head_merge if layer_idx == gat_layers - 1 else gat_head_merge
            self.layers.append(
                GATLayer(
                    input_dim=in_dim,
                    output_dim=out_dim,
                    heads=gat_heads,
                    dropout=gat_dropout,
                    head_merge=head_merge,
                )
            )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        node_features = x.transpose(1, 2)
        node_features = self.input_dropout(self.input_proj(node_features))

        for idx, layer in enumerate(self.layers):
            node_features = layer(node_features, adjacency)
            if idx < len(self.layers) - 1:
                node_features = F.elu(node_features)

        return node_features


class FFTEncoder(nn.Module):
    def __init__(
        self,
        freq_bins: int,
        fft_hidden: int,
        fft_n_heads: int,
        fft_n_layers: int,
        fft_dim_feedforward: int,
        fft_dropout: float,
        fft_readout: str,
    ) -> None:
        super().__init__()
        if fft_readout not in {"mean", "cls", "last"}:
            raise ValueError(f"Unsupported fft_readout: {fft_readout}")

        self.fft_readout = fft_readout
        self.input_proj = nn.Linear(1, fft_hidden)
        token_count = freq_bins + (1 if fft_readout == "cls" else 0)
        self.pos_embedding = nn.Parameter(torch.zeros(1, token_count, fft_hidden))
        self.cls_token = None
        if fft_readout == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, fft_hidden))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fft_hidden,
            nhead=fft_n_heads,
            dim_feedforward=fft_dim_feedforward,
            dropout=fft_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=fft_n_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(fft_hidden, fft_hidden),
            nn.ReLU(),
            nn.Dropout(fft_dropout),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, freq_bins = spectrum.shape
        tokens = spectrum.reshape(batch_size * num_nodes, freq_bins, 1)
        tokens = self.input_proj(tokens)

        if self.cls_token is not None:
            cls = self.cls_token.expand(batch_size * num_nodes, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        tokens = tokens + self.pos_embedding[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens)

        if self.fft_readout == "cls":
            pooled = encoded[:, 0, :]
        elif self.fft_readout == "last":
            pooled = encoded[:, -1, :]
        else:
            pooled = encoded.mean(dim=1)

        pooled = self.output_proj(pooled)
        return pooled.reshape(batch_size, num_nodes, -1)


class FrequencyBranch(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        hidden_dim: int,
        fft_hidden: int,
        fft_n_heads: int,
        fft_n_layers: int,
        fft_dim_feedforward: int,
        fft_dropout: float,
        fft_readout: str,
    ) -> None:
        super().__init__()
        freq_bins = (sequence_length // 2) + 1
        self.mag_encoder = FFTEncoder(
            freq_bins=freq_bins,
            fft_hidden=fft_hidden,
            fft_n_heads=fft_n_heads,
            fft_n_layers=fft_n_layers,
            fft_dim_feedforward=fft_dim_feedforward,
            fft_dropout=fft_dropout,
            fft_readout=fft_readout,
        )
        self.phase_encoder = FFTEncoder(
            freq_bins=freq_bins,
            fft_hidden=fft_hidden,
            fft_n_heads=fft_n_heads,
            fft_n_layers=fft_n_layers,
            fft_dim_feedforward=fft_dim_feedforward,
            fft_dropout=fft_dropout,
            fft_readout=fft_readout,
        )
        self.fuse = nn.Sequential(
            nn.Linear(fft_hidden * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(fft_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        node_series = x.transpose(1, 2)
        spectrum = torch.fft.rfft(node_series, dim=-1)
        magnitude = torch.abs(spectrum).float()
        phase = torch.angle(spectrum).float()

        mag_repr = self.mag_encoder(magnitude)
        phase_repr = self.phase_encoder(phase)
        return self.fuse(torch.cat([mag_repr, phase_repr], dim=-1))


class TrafficModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        adjacency: torch.Tensor,
        horizon: int = 1,
        hidden_dim: int = 64,
        rnn_type: str = "bilstm",
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
        bilstm_input_proj_dim: int = 16,
        readout: str = "mean",
        fft_hidden: int = 64,
        fft_n_heads: int = 4,
        fft_n_layers: int = 2,
        fft_dim_feedforward: int = 256,
        fft_dropout: float = 0.1,
        fft_readout: str = "mean",
        gat_hidden: int = 64,
        gat_heads: int = 4,
        gat_layers: int = 2,
        gat_dropout: float = 0.1,
        gat_input_proj_dim: int = 32,
        gat_head_merge: str = "mean",
        gat_final_head_merge: str = "mean",
    ) -> None:
        super().__init__()
        if readout not in {"mean", "attention", "gated"}:
            raise ValueError(f"Unsupported readout: {readout}")

        adjacency_tensor = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency_tensor.shape != (input_dim, input_dim):
            raise ValueError(
                f"Adjacency shape {tuple(adjacency_tensor.shape)} does not match input_dim={input_dim}"
            )

        self.num_nodes = input_dim
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.readout = readout

        self.register_buffer("adjacency", adjacency_tensor)

        self.temporal_branch = TemporalBranch(
            hidden_dim=hidden_dim,
            rnn_type=rnn_type,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            lstm_dropout=lstm_dropout,
            bilstm_input_proj_dim=bilstm_input_proj_dim,
        )
        self.spatial_branch = SpatialBranch(
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            gat_hidden=gat_hidden,
            gat_heads=gat_heads,
            gat_layers=gat_layers,
            gat_dropout=gat_dropout,
            gat_input_proj_dim=gat_input_proj_dim,
            gat_head_merge=gat_head_merge,
            gat_final_head_merge=gat_final_head_merge,
        )
        self.frequency_branch = FrequencyBranch(
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            fft_hidden=fft_hidden,
            fft_n_heads=fft_n_heads,
            fft_n_layers=fft_n_layers,
            fft_dim_feedforward=fft_dim_feedforward,
            fft_dropout=fft_dropout,
            fft_readout=fft_readout,
        )

        head_dropout = max(lstm_dropout, gat_dropout, fft_dropout)
        self.node_head_hidden = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
        )
        self.node_head_out = nn.Linear(hidden_dim, horizon)

        self.attention_query = None
        self.gated_logits = None
        if readout == "attention":
            self.attention_query = nn.Parameter(torch.empty(hidden_dim))
            nn.init.uniform_(self.attention_query, -1.0 / math.sqrt(hidden_dim), 1.0 / math.sqrt(hidden_dim))
        elif readout == "gated":
            self.gated_logits = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x.ndim == 3, got {x.ndim}")
        if x.shape[1] != self.sequence_length:
            raise ValueError(
                f"Input seq_len={x.shape[1]} does not match configured sequence_length={self.sequence_length}"
            )
        if x.shape[2] != self.num_nodes:
            raise ValueError(f"Input num_nodes={x.shape[2]} does not match configured {self.num_nodes}")

        temp_repr = self.temporal_branch(x)
        spatial_repr = self.spatial_branch(x, self.adjacency)
        freq_repr = self.frequency_branch(x)

        fused = torch.cat([temp_repr, spatial_repr, freq_repr], dim=-1)
        node_hidden = self.node_head_hidden(fused)
        node_pred = self.node_head_out(node_hidden)

        if self.readout == "mean":
            return node_pred.mean(dim=1)

        if self.readout == "attention":
            scores = torch.matmul(node_hidden, self.attention_query)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            return torch.sum(node_pred * weights, dim=1)

        weights = torch.softmax(self.gated_logits, dim=0).view(1, self.num_nodes, 1)
        return torch.sum(node_pred * weights, dim=1)
