"""
flow_model.py
-------------
Full macro-conditional normalizing flow: TFT encoder + MAF decoder.

    macro_seq → [TFT Encoder] → h_t → [MAF Decoder] → log p(X_t | h_t)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.maf import MAFlow
from src.models.tft import TemporalFusionTransformer


class ConditionalNormalizingFlow(nn.Module):
    """
    End-to-end macro-conditional normalizing flow for portfolio risk.

    The TFT reads a macro history sequence and produces a context vector h_t.
    The MAF uses h_t to condition its affine transformations, adapting the
    return distribution shape to the current macroeconomic regime.

    Training objective: minimize NLL = -E[log p(X_t | h_t)].
    Inference: sample from the flow conditioned on h_t for Monte Carlo VaR/ES.

    Parameters
    ----------
    num_macro_features : int
        Number of macro/market features per time step (TFT input width).
    num_assets : int
        Number of assets (flow output dimension D).
    tft_d_model : int
        Hidden dimension of the TFT encoder.
    tft_n_heads : int
        Number of attention heads in the TFT.
    tft_n_lstm_layers : int
        Number of LSTM layers in the TFT.
    flow_n_layers : int
        Number of MAFLayer steps in the flow.
    flow_hidden_dim : int
        Hidden dimension inside each MADE network.
    flow_n_hidden : int
        Number of hidden layers inside each MADE.
    dropout : float
        Dropout rate applied in both TFT and flow.
    """

    def __init__(
        self,
        num_macro_features: int,
        num_assets: int,
        tft_d_model: int = 128,
        tft_n_heads: int = 4,
        tft_n_lstm_layers: int = 2,
        flow_n_layers: int = 8,
        flow_hidden_dim: int = 128,
        flow_n_hidden: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_macro_features = num_macro_features
        self.num_assets = num_assets
        self.tft_d_model = tft_d_model

        self.tft = TemporalFusionTransformer(
            num_features=num_macro_features,
            d_model=tft_d_model,
            n_heads=tft_n_heads,
            n_lstm_layers=tft_n_lstm_layers,
            dropout=dropout,
        )
        self.flow = MAFlow(
            dim=num_assets,
            n_layers=flow_n_layers,
            hidden_dim=flow_hidden_dim,
            n_hidden=flow_n_hidden,
            context_dim=tft_d_model,
            use_batch_norm=True,
        )

    def encode(self, macro_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode macro history into context vector h_t.

        Parameters
        ----------
        macro_seq : Tensor of shape (batch, seq_len, num_macro_features)

        Returns
        -------
        h_t : Tensor of shape (batch, tft_d_model)
        var_weights : Tensor of shape (batch, seq_len, num_macro_features)
        """
        return self.tft(macro_seq)

    def log_prob(
        self,
        x: torch.Tensor,
        macro_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log p(X_t | M_{<t}) for a batch of (returns, macro_seq) pairs.

        Parameters
        ----------
        x : Tensor of shape (batch, num_assets)
            Observed asset log returns on day t.
        macro_seq : Tensor of shape (batch, seq_len, num_macro_features)
            Historical macro features up to (but not including) day t.

        Returns
        -------
        log_prob : Tensor of shape (batch,)
        """
        h_t, _ = self.tft(macro_seq)
        return self.flow.log_prob(x, context=h_t)

    def forward(
        self,
        x: torch.Tensor,
        macro_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute NLL loss and variable importance weights.

        Parameters
        ----------
        x : Tensor of shape (batch, num_assets)
        macro_seq : Tensor of shape (batch, seq_len, num_macro_features)

        Returns
        -------
        nll : Tensor scalar — mean NLL over the batch.
        var_weights : Tensor of shape (batch, seq_len, num_macro_features)
        """
        h_t, var_weights = self.tft(macro_seq)
        nll = -self.flow.log_prob(x, context=h_t).mean()
        return nll, var_weights

    @torch.no_grad()
    def sample(
        self,
        macro_seq: torch.Tensor,
        n_samples: int = 10_000,
    ) -> torch.Tensor:
        """
        Draw Monte Carlo samples from p(X_t | M_{<t}).

        Parameters
        ----------
        macro_seq : Tensor of shape (1, seq_len, num_macro_features)
            Macro sequence for a single day t.
        n_samples : int
            Number of Monte Carlo samples to draw.

        Returns
        -------
        samples : Tensor of shape (n_samples, num_assets)
        """
        self.eval()
        h_t, _ = self.tft(macro_seq)
        return self.flow.sample(n_samples, context=h_t)

    def get_variable_importance(
        self,
        macro_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract averaged variable importance weights for interpretability.

        Parameters
        ----------
        macro_seq : Tensor of shape (batch, seq_len, num_macro_features)

        Returns
        -------
        importance : Tensor of shape (num_macro_features,)
            Mean importance weight per feature across batch and time.
        """
        self.eval()
        with torch.no_grad():
            _, var_weights = self.tft(macro_seq)
            return var_weights.mean(dim=(0, 1))

    def count_parameters(self) -> Dict[str, int]:
        """Return trainable parameter counts per component."""
        tft_params = sum(p.numel() for p in self.tft.parameters() if p.requires_grad)
        flow_params = sum(p.numel() for p in self.flow.parameters() if p.requires_grad)
        return {"tft": tft_params, "flow": flow_params, "total": tft_params + flow_params}


if __name__ == "__main__":
    B, T, F, D = 8, 63, 10, 3

    model = ConditionalNormalizingFlow(
        num_macro_features=F,
        num_assets=D,
        tft_d_model=64,
        flow_n_layers=4,
        flow_hidden_dim=64,
    )

    macro_seq = torch.randn(B, T, F)
    returns = torch.randn(B, D)

    nll, weights = model(returns, macro_seq)
    print(f"NLL:           {nll.item():.4f}")
    print(f"Weights shape: {weights.shape}")

    samples = model.sample(macro_seq[:1], n_samples=100)
    print(f"Samples shape: {samples.shape}")

    params = model.count_parameters()
    print(f"Parameters: TFT={params['tft']:,}, Flow={params['flow']:,}, Total={params['total']:,}")