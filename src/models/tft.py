"""
tft.py
------
Temporal Fusion Transformer (TFT) — macro regime encoder.

Adapted from Lim et al. (2021), "Temporal Fusion Transformers for
Interpretable Multi-horizon Time Series Forecasting."

Reads a sequence of macroeconomic features over the past seq_len trading
days and compresses the history into a fixed-size context vector h_t used
to condition the normalizing flow decoder.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN).

    Applies a gated skip-connection that allows the network to learn whether
    to apply or bypass a non-linear transformation.

    Parameters
    ----------
    input_size : int
        Dimension of the input tensor.
    hidden_size : int
        Dimension of the hidden layer.
    output_size : int
        Dimension of the output tensor.
    dropout : float
        Dropout rate applied after the ELU activation.
    context_size : int, optional
        If provided, an external context vector is injected into the hidden layer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.context_fc = nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        self.fc2 = nn.Linear(hidden_size, output_size * 2)
        self.skip_fc = nn.Linear(input_size, output_size) if input_size != output_size else None
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (..., input_size)
        context : Tensor of shape (..., context_size), optional

        Returns
        -------
        Tensor of shape (..., output_size)
        """
        h = self.fc1(x)
        if context is not None and self.context_fc is not None:
            h = h + self.context_fc(context)
        h = self.dropout(F.elu(h))
        h = self.fc2(h)

        h1, h2 = h.chunk(2, dim=-1)
        h = h1 * torch.sigmoid(h2)

        skip = self.skip_fc(x) if self.skip_fc is not None else x
        return self.layer_norm(h + skip)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN).

    Learns soft attention weights over input variables, providing interpretable
    feature importance scores.

    Parameters
    ----------
    input_size : int
        Embedding dimension of each input variable.
    num_vars : int
        Number of input variables.
    hidden_size : int
        Hidden dimension for internal GRNs.
    dropout : float
        Dropout rate.
    context_size : int, optional
        If provided, context is injected into the selection GRN.
    """

    def __init__(
        self,
        input_size: int,
        num_vars: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.num_vars = num_vars

        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
            for _ in range(num_vars)
        ])
        self.selection_grn = GatedResidualNetwork(
            input_size=input_size * num_vars,
            hidden_size=hidden_size,
            output_size=num_vars,
            dropout=dropout,
            context_size=context_size,
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, num_vars, input_size)

        Returns
        -------
        output : Tensor of shape (batch, seq_len, hidden_size)
        weights : Tensor of shape (batch, seq_len, num_vars)
        """
        B, T, V, d = x.shape

        var_outputs = torch.stack(
            [grn(x[:, :, i, :]) for i, grn in enumerate(self.var_grns)],
            dim=2,
        )  # (B, T, V, hidden)

        weights = self.softmax(
            self.selection_grn(x.reshape(B, T, V * d), context)
        )  # (B, T, V)

        output = (var_outputs * weights.unsqueeze(-1)).sum(dim=2)  # (B, T, hidden)
        return output, weights


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT) — macro regime encoder.

    Encodes a macroeconomic feature sequence of shape (batch, seq_len, num_features)
    into a single context vector h_t of shape (batch, d_model).

    Parameters
    ----------
    num_features : int
        Number of macro/market features per time step.
    d_model : int
        Hidden dimension throughout the TFT.
    n_heads : int
        Number of attention heads.
    n_lstm_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_lstm_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model

        self.input_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])
        self.vsn = VariableSelectionNetwork(
            input_size=d_model,
            num_vars=num_features,
            hidden_size=d_model,
            dropout=dropout,
        )
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_lstm_layers,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.post_lstm_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_layer_norm = nn.LayerNorm(d_model)
        self.post_attn_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.final_grn = GatedResidualNetwork(d_model, d_model * 4, d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a macroeconomic sequence into context vector h_t.

        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, num_features)

        Returns
        -------
        h_t : Tensor of shape (batch, d_model)
            Compressed macro regime vector (last time step).
        var_weights : Tensor of shape (batch, seq_len, num_features)
            Variable importance weights.
        """
        B, T, F = x.shape
        assert F == self.num_features, f"Expected {self.num_features} features, got {F}"

        # 1. Embed each feature independently → (B, T, num_features, d_model)
        embedded = torch.stack(
            [emb(x[:, :, i].unsqueeze(-1)) for i, emb in enumerate(self.input_embeddings)],
            dim=2,
        )

        # 2. Variable selection → (B, T, d_model)
        vsn_out, var_weights = self.vsn(embedded)

        # 3. LSTM encoder → (B, T, d_model)
        lstm_out, _ = self.lstm(vsn_out)

        # 4. Post-LSTM GRN with skip connection
        lstm_filtered = self.post_lstm_grn(lstm_out + vsn_out)

        # 5. Multi-head self-attention + residual → (B, T, d_model)
        attn_out, _ = self.attention(lstm_filtered, lstm_filtered, lstm_filtered)
        attn_out = self.attention_layer_norm(attn_out + lstm_filtered)

        # 6. Post-attention GRN
        attn_filtered = self.post_attn_grn(attn_out)

        # 7. Final feedforward GRN
        out = self.final_grn(attn_filtered)

        # 8. Last time step → context vector h_t
        h_t = out[:, -1, :]  # (B, d_model)

        return h_t, var_weights


if __name__ == "__main__":
    B, T, F = 8, 63, 10
    model = TemporalFusionTransformer(num_features=F, d_model=64, n_heads=4)
    x = torch.randn(B, T, F)
    h_t, weights = model(x)
    print(f"h_t shape:     {h_t.shape}")
    print(f"weights shape: {weights.shape}")
    print(f"Parameters:    {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")