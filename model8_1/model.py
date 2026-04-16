from __future__ import annotations

import torch
import torch.nn as nn


class TrafficModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        horizon: int = 1,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 512,
        vit_dropout: float = 0.2,
        rnn_type: str = "lstm",
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if rnn_type not in {"lstm", "bilstm"}:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        self.horizon = horizon
        self.sequence_length = sequence_length
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, sequence_length, d_model))

        bidirectional = rnn_type == "bilstm"
        rnn_dropout = lstm_dropout if lstm_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
        )

        rnn_output_dim = lstm_hidden * (2 if bidirectional else 1)
        head_hidden = max(1, rnn_output_dim // 2)
        self.head = nn.Sequential(
            nn.Linear(rnn_output_dim, rnn_output_dim),
            nn.ReLU(),
            nn.Dropout(lstm_dropout),
            nn.Linear(rnn_output_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, horizon),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        seq_len = x.shape[1]
        if seq_len > self.sequence_length:
            raise ValueError(
                f"Input seq_len={seq_len} exceeds configured sequence_length={self.sequence_length}"
            )

        projected = self.input_projection(x)
        projected = projected + self.pos_embedding[:, :seq_len, :]

        rnn_out, _ = self.rnn(projected)
        last_step = rnn_out[:, -1, :]
        output = self.head(last_step)
        return output
