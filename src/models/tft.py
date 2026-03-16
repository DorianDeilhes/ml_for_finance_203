"""
tft.py
------
Temporal Fusion Transformer (TFT) — Macro-Regime Encoder.

Architecture adapted from Lim et al. (2021) "Temporal Fusion Transformers for
Interpretable Multi-horizon Time Series Forecasting."

The TFT reads a sequence of macroeconomic features over the past `seq_len` trading
days and compresses the entire history into a fixed-size context vector h_t that
captures the current macroeconomic regime. This vector is then used to condition
the Normalizing Flow decoder.

Key components:
    GatedResidualNetwork (GRN) : Learns to suppress irrelevant macro signals.
    VariableSelectionNetwork  : Provides soft feature weighting (explainable AI).
    LSTMEncoder               : Captures sequential temporal dependencies.
    MultiHeadAttention        : Long-range temporal attention.
    TemporalFusionTransformer : Full encoder, outputs h_t.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Gated Residual Network
# ─────────────────────────────────────────────────────────────────────────────

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN).

    The GRN provides gated skip-connections that allow the network to learn
    whether to use or suppress a transformation. This is critical for filtering
    out irrelevant macroeconomic noise.

    Architecture:
        x → Linear → ELU → Linear → [GLU gate] → LayerNorm + skip
        (optional external context injected between the two linears)

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
        If provided, an additional context vector is injected into the GRN.
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

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Primary transformation: input → hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Optional context injection
        self.context_fc = nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        # Second transformation: hidden → 2 * output (for GLU)
        self.fc2 = nn.Linear(hidden_size, output_size * 2)

        # Skip connection (projects input to output_size if dimensions differ)
        if input_size != output_size:
            self.skip_fc = nn.Linear(input_size, output_size)
        else:
            self.skip_fc = None

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
        context : Optional Tensor of shape (..., context_size)

        Returns
        -------
        Tensor of shape (..., output_size)
        """
        # Primary path
        h = self.fc1(x)
        if context is not None and self.context_fc is not None:
            h = h + self.context_fc(context)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.fc2(h)

        # Gated Linear Unit (GLU): splits output into value and gate halves
        h1, h2 = h.chunk(2, dim=-1)
        h = h1 * torch.sigmoid(h2)

        # Skip connection + LayerNorm
        skip = self.skip_fc(x) if self.skip_fc is not None else x
        return self.layer_norm(h + skip)


# ─────────────────────────────────────────────────────────────────────────────
# Variable Selection Network
# ─────────────────────────────────────────────────────────────────────────────

class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN).

    Learns soft attention weights over input variables, providing interpretable
    feature importance scores. Different macro variables will be weighted
    differently depending on the economic regime.

    For an input of V variables each with embedding dimension d:
    - Each variable is individually processed through a GRN.
    - A combined representation is processed through another GRN to produce
      softmax weights v_1, ..., v_V.
    - The output is the weighted sum: sum_i(v_i * grn_i(x_i)).

    Parameters
    ----------
    input_size : int
        Dimension of each input variable's embedding.
    num_vars : int
        Number of input variables (V).
    hidden_size : int
        Hidden dimension for GRNs.
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

        # Per-variable GRNs
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
            for _ in range(num_vars)
        ])

        # Selection GRN: processes the flattened concatenation of all variables
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
            OR (batch, num_vars, input_size) for static variables

        Returns
        -------
        output : Tensor of shape (batch, seq_len, hidden_size)
        weights : Tensor of shape (batch, seq_len, num_vars)  [interpretable!]
        """
        # x: (B, T, V, d)
        B, T, V, d = x.shape

        # Compute per-variable GRN transformations
        var_outputs = []
        for i, grn in enumerate(self.var_grns):
            var_outputs.append(grn(x[:, :, i, :]))  # (B, T, hidden)
        var_outputs = torch.stack(var_outputs, dim=2)  # (B, T, V, hidden)

        # Compute selection weights using flattened input
        flat = x.reshape(B, T, V * d)              # (B, T, V*d)
        weights_logits = self.selection_grn(flat, context)  # (B, T, V)
        weights = self.softmax(weights_logits)      # (B, T, V)

        # Weighted combination
        weights_expanded = weights.unsqueeze(-1)    # (B, T, V, 1)
        output = (var_outputs * weights_expanded).sum(dim=2)  # (B, T, hidden)

        return output, weights


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Fusion Transformer
# ─────────────────────────────────────────────────────────────────────────────

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT) — Macro Regime Encoder.

    Reads a sequence of macroeconomic features of shape (batch, seq_len, num_features)
    and outputs a single context vector h_t of shape (batch, d_model) representing
    the compressed macro regime state.

    Architecture overview:
        1. Input Embedding: Linear projection per feature → embeddings
        2. Variable Selection Network: Soft-weighted feature selection
        3. LSTM Encoder: Sequential pattern capture
        4. GRN skip connection: Gated filter
        5. Multi-Head Self-Attention: Long-range temporal dependencies
        6. GRN post-attention: Further gated filtering
        7. Position-wise FF (GRN): Final representation
        8. Pooling: take the last time step as h_t

    Parameters
    ----------
    num_features : int
        Number of macro/market features per time step.
    d_model : int
        Hidden dimension throughout the TFT (embedding size).
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

        # ── 1. Input Embedding (project each scalar feature to d_model) ──────
        # Each feature gets its own linear layer
        self.input_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])

        # ── 2. Variable Selection Network ─────────────────────────────────────
        self.vsn = VariableSelectionNetwork(
            input_size=d_model,
            num_vars=num_features,
            hidden_size=d_model,
            dropout=dropout,
        )

        # ── 3. LSTM Encoder ───────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_lstm_layers,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        # ── 4. GRN skip (post-LSTM) ───────────────────────────────────────────
        self.post_lstm_grn = GatedResidualNetwork(
            input_size=d_model,
            hidden_size=d_model,
            output_size=d_model,
            dropout=dropout,
        )

        # ── 5. Multi-Head Attention ───────────────────────────────────────────
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_layer_norm = nn.LayerNorm(d_model)

        # ── 6. GRN post-attention ─────────────────────────────────────────────
        self.post_attn_grn = GatedResidualNetwork(
            input_size=d_model,
            hidden_size=d_model,
            output_size=d_model,
            dropout=dropout,
        )

        # ── 7. Final feedforward GRN ──────────────────────────────────────────
        self.final_grn = GatedResidualNetwork(
            input_size=d_model,
            hidden_size=d_model * 4,
            output_size=d_model,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode macroeconomic sequence into context vector h_t.

        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, num_features)
            Scaled macroeconomic feature sequences.

        Returns
        -------
        h_t : Tensor of shape (batch, d_model)
            Compressed macro regime context vector (last time step).
        var_weights : Tensor of shape (batch, seq_len, num_features)
            Variable importance weights (interpretable).
        """
        B, T, F = x.shape
        assert F == self.num_features, (
            f"Expected {self.num_features} features, got {F}"
        )

        # ── 1. Embed each feature dimension independently ─────────────────────
        embedded = []
        for i, emb_layer in enumerate(self.input_embeddings):
            # x[:, :, i] → (B, T, 1) → (B, T, d_model)
            feat = x[:, :, i].unsqueeze(-1)
            embedded.append(emb_layer(feat))
        # Stack → (B, T, num_features, d_model)
        embedded = torch.stack(embedded, dim=2)

        # ── 2. Variable Selection ─────────────────────────────────────────────
        vsn_out, var_weights = self.vsn(embedded)  # (B, T, d_model), (B, T, F)

        # ── 3. LSTM Encoder ───────────────────────────────────────────────────
        lstm_out, _ = self.lstm(vsn_out)  # (B, T, d_model)

        # ── 4. GRN skip (post-LSTM) ───────────────────────────────────────────
        lstm_filtered = self.post_lstm_grn(lstm_out + vsn_out)  # skip connection

        # ── 5. Multi-Head Self-Attention ──────────────────────────────────────
        attn_out, _ = self.attention(
            lstm_filtered, lstm_filtered, lstm_filtered
        )  # (B, T, d_model)
        # Add + Norm (residual)
        attn_out = self.attention_layer_norm(attn_out + lstm_filtered)

        # ── 6. GRN post-attention ─────────────────────────────────────────────
        attn_filtered = self.post_attn_grn(attn_out)

        # ── 7. Final feedforward ──────────────────────────────────────────────
        out = self.final_grn(attn_filtered)  # (B, T, d_model)

        # ── 8. Take last time step as the context vector h_t ─────────────────
        h_t = out[:, -1, :]  # (B, d_model)

        return h_t, var_weights


if __name__ == "__main__":
    # Quick sanity check
    B, T, F = 8, 63, 10
    model = TemporalFusionTransformer(num_features=F, d_model=64, n_heads=4)
    x = torch.randn(B, T, F)
    h_t, weights = model(x)
    print(f"h_t shape:      {h_t.shape}")       # Expected: (8, 64)
    print(f"weights shape:  {weights.shape}")   # Expected: (8, 63, 10)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
