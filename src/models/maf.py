"""
maf.py
------
Masked Autoregressive Flow (MAF) for Conditional Density Estimation.

A Normalizing Flow transforms a simple base distribution (e.g., a standard
Gaussian) into a complex, fat-tailed distribution via a sequence of invertible
neural network layers.

The key innovation of the MAF is the use of the MADE (Masked Autoencoder for
Distribution Estimation) to enforce the autoregressive property:

    x_i only depends on x_1, ..., x_{i-1}

This ensures that the Jacobian of the transformation is strictly lower-triangular,
making its determinant computable in O(D) rather than O(D^3):

    det(J) = product of diagonal elements = product of exp(alpha_i)

Architecture:
    MADE:     Masked autoregressive network outputting (alpha, mu) for each dim.
    MAFLayer: Affine transform: x_i = z_i * exp(alpha_i(x_{<i}; h)) + mu_i(x_{<i}; h)
    MAFlow:   Stack of N MAFLayers with ordering reversal between layers.

References:
    - Papamakarios et al. (2017): "Masked Autoregressive Flow for Density Estimation"
    - Germain et al. (2015): "MADE: Masked Autoencoder for Distribution Estimation"
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Masked Linear Layer
# ─────────────────────────────────────────────────────────────────────────────

class MaskedLinear(nn.Linear):
    """
    A linear layer with a binary mask applied to the weight matrix.
    Used to enforce the autoregressive property in MADE.

    The mask ensures that output unit i can only receive information from
    input units j where mask[i, j] = 1 (corresponding to x_{<i}).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        # Register mask as a buffer (not a parameter — not updated by optimizer)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask: torch.Tensor) -> None:
        """Set the autoregressive mask."""
        self.mask.data.copy_(mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply mask to weight matrix before computing linear transformation
        return F.linear(x, self.weight * self.mask, self.bias)


# ─────────────────────────────────────────────────────────────────────────────
# MADE (Masked Autoencoder for Distribution Estimation)
# ─────────────────────────────────────────────────────────────────────────────

class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation (MADE).

    A feedforward network with strategically masked weight matrices to enforce
    the autoregressive ordering property: the output for dimension i depends
    only on inputs 1, ..., i-1.

    When conditioned on an external context h_t (from the TFT), the context
    is concatenated to the input before the first masked layer.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input (number of assets, D).
    hidden_dim : int
        Width of hidden layers.
    n_hidden : int
        Number of hidden layers.
    context_dim : int, optional
        Dimensionality of the conditioning context vector h_t.
    activation : str
        Activation function ('relu', 'tanh', 'elu').
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        context_dim: Optional[int] = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.input_dim   = input_dim
        self.hidden_dim  = hidden_dim
        self.n_hidden    = n_hidden
        self.context_dim = context_dim

        # Context projection: h_t → context_dim (if provided)
        self.context_proj: Optional[nn.Linear] = None
        effective_input = input_dim
        if context_dim is not None:
            # Project context to same size as input for concatenation
            self.context_proj = nn.Linear(context_dim, context_dim)
            effective_input = input_dim + context_dim

        # Build masked layers
        # Sizes: [effective_input, hidden, ..., hidden, 2*input_dim]
        # The final layer outputs (alpha, mu) for each of D dimensions
        layer_sizes = [effective_input] + [hidden_dim] * n_hidden + [input_dim * 2]

        self.layers = nn.ModuleList()
        for in_sz, out_sz in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(MaskedLinear(in_sz, out_sz))

        self.activation = {
            "relu": F.relu,
            "tanh": torch.tanh,
            "elu":  F.elu,
        }[activation]

        # Assign and set masks
        self._setup_masks(effective_input, hidden_dim, input_dim)

    def _setup_masks(
        self,
        effective_input: int,
        hidden_size: int,
        output_size: int,
    ) -> None:
        """
        Construct autoregressive masks for each MADE layer.

        Each unit in each layer is assigned an ordering number m(k) between
        1 and D-1. Connections are only allowed from units with lower order
        to units with equal-or-higher order.

        For the output layer, we enforce STRICT inequality (output_i only
        sees input with order < i), giving us the lower-triangular Jacobian.
        """
        input_dim = self.input_dim
        context_dim = self.context_dim if self.context_dim is not None else 0

        # Assign degree to each input unit
        # Context units get degree 0 (always visible to all)
        if context_dim > 0:
            m_input = np.concatenate([
                np.zeros(context_dim, dtype=int),        # context: degree 0
                np.arange(1, input_dim + 1, dtype=int),  # x dims: degree 1..D
            ])
        else:
            m_input = np.arange(1, input_dim + 1, dtype=int)

        # Hidden layer degrees: randomly chosen in [1, D-1] for each unit
        rng = np.random.default_rng(seed=42)
        m_hidden = [rng.integers(low=1, high=input_dim, size=self.hidden_dim, endpoint=False)
                    for _ in range(self.n_hidden)]

        # Output layer degrees: first D outputs are alpha, next D are mu.
        # Because forward() does `alpha, mu = h.chunk(2, dim=-1)`, degrees must be
        # [0..D-1, 0..D-1] so alpha_i and mu_i share the same autoregressive order.
        m_output = np.concatenate([
            np.arange(0, input_dim),
            np.arange(0, input_dim),
        ])

        # Build mask for each layer
        all_m = [m_input] + m_hidden + [m_output]
        actual_layer_idx = 0

        for i, layer in enumerate(self.layers):
            m_prev = all_m[i]
            m_curr = all_m[i + 1]

            # Hidden layer: m_curr >= m_prev (non-strict)
            # Output layer: m_curr >= m_prev (but output ordering is 0..D-1 vs input 1..D)
            # This naturally gives strict lower-triangular Jacobian for outputs
            is_output = (i == len(self.layers) - 1)
            if is_output:
                # Strict: output i (degree i-1) only sees input j (degree ≤ i-1)
                mask = torch.tensor(
                    (m_curr[:, None] >= m_prev[None, :]).astype(np.float32)
                )
            else:
                mask = torch.tensor(
                    (m_curr[:, None] >= m_prev[None, :]).astype(np.float32)
                )
            layer.set_mask(mask)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute autoregressive parameters (alpha, mu) for each dimension.

        Parameters
        ----------
        x : Tensor of shape (batch, D)
            The input values (used as conditioning for subsequent dimensions).
        context : Tensor of shape (batch, context_dim), optional
            The macro context vector h_t from the TFT.

        Returns
        -------
        alpha : Tensor of shape (batch, D)
            Log-scale parameters for the affine transform.
        mu : Tensor of shape (batch, D)
            Shift parameters for the affine transform.
        """
        # Concatenate context if provided
        if context is not None and self.context_proj is not None:
            ctx = self.context_proj(context)
            h = torch.cat([ctx, x], dim=-1)
        else:
            h = x

        # Forward pass through masked layers
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)

        # Output: split into alpha and mu (both shape: batch, D)
        alpha, mu = h.chunk(2, dim=-1)

        # Smoothly bound alpha to keep exp(alpha) numerically stable.
        alpha = 3.0 * torch.tanh(alpha / 3.0)

        return alpha, mu


# ─────────────────────────────────────────────────────────────────────────────
# MAF Layer (Single Autoregressive Flow Step)
# ─────────────────────────────────────────────────────────────────────────────

class MAFLayer(nn.Module):
    """
    A single step of the Masked Autoregressive Flow.

    Implements the affine transformation:
        Forward (data → noise):   z_i = (x_i - mu_i(x_{<i}; h)) * exp(-alpha_i(x_{<i}; h))
        Inverse (noise → data):   x_i = z_i * exp(alpha_i) + mu_i

    The log Jacobian determinant (needed for NLL) is:
        log|det J| = -sum_i(alpha_i)    [triangular, O(D) computation]

    Note: The FORWARD direction (x → z) is used during TRAINING (density estimation).
          The INVERSE direction (z → x) is used during SAMPLING (Monte Carlo).

    Parameters
    ----------
    dim : int
        Dimensionality of the data (D = number of assets).
    hidden_dim : int
        Hidden dimension in the MADE network.
    n_hidden : int
        Number of hidden layers in MADE.
    context_dim : int, optional
        Dimension of conditioning context h_t.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.made = MADE(
            input_dim=dim,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            context_dim=context_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map data x → noise z (used during training for NLL computation).

        Parameters
        ----------
        x : Tensor of shape (batch, D)
        context : Tensor of shape (batch, context_dim), optional

        Returns
        -------
        z : Tensor of shape (batch, D)  — noise in base distribution space
        log_det_J : Tensor of shape (batch,)  — log |det Jacobian|
        """
        alpha, mu = self.made(x, context)
        # x → z: invert the affine transform
        z = (x - mu) * torch.exp(-alpha)
        # log|det J| = sum of -alpha over dimensions (triangular Jacobian)
        log_det_J = -alpha.sum(dim=-1)
        return z, log_det_J

    @torch.no_grad()
    def inverse(
        self,
        z: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Map noise z → data x (used during Monte Carlo sampling).

        The inverse pass is SEQUENTIAL (autoregressive) — x_i must be
        computed before x_{i+1}. This takes O(D) MADE evaluations.

        Parameters
        ----------
        z : Tensor of shape (batch, D)
        context : Tensor of shape (batch, context_dim), optional

        Returns
        -------
        x : Tensor of shape (batch, D)
        """
        D = z.shape[-1]
        x = torch.zeros_like(z)
        for i in range(D):
            alpha, mu = self.made(x, context)
            # Recover x_i from z_i using the affine transform
            x[:, i] = z[:, i] * torch.exp(alpha[:, i]) + mu[:, i]
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Batch Normalization for Flows
# ─────────────────────────────────────────────────────────────────────────────

class FlowBatchNorm(nn.Module):
    """
    Invertible Batch Normalization for Normalizing Flows.

    Standard BatchNorm is not directly invertible during inference. This
    implementation uses running statistics so the forward and inverse passes
    are always consistent.

    This stabilizes training by normalizing activations between flow layers.
    """

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.zeros(dim))     # log scale
        self.beta  = nn.Parameter(torch.zeros(dim))     # shift
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var",  torch.ones(dim))

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize x → z, return log_det_J."""
        if self.training:
            mean = x.mean(0)
            var  = x.var(0, unbiased=False)
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var.detach()
        else:
            mean = self.running_mean
            var  = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        z = x_hat * torch.exp(self.gamma) + self.beta

        # log|det J| = sum(gamma) - 0.5 * sum(log(var + eps))
        log_det_J = (self.gamma - 0.5 * torch.log(var + self.eps)).sum()
        log_det_J = log_det_J.expand(x.shape[0])

        return z, log_det_J

    @torch.no_grad()
    def inverse(
        self,
        z: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Invert BatchNorm: z → x."""
        x_hat = (z - self.beta) * torch.exp(-self.gamma)
        x = x_hat * torch.sqrt(self.running_var + self.eps) + self.running_mean
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full Masked Autoregressive Flow (stacked MAF layers)
# ─────────────────────────────────────────────────────────────────────────────

class MAFlow(nn.Module):
    """
    Full Masked Autoregressive Flow: a stack of invertible MAF layers.

    Design choices:
    - Between each MAFLayer, we reverse the ordering of dimensions.
      This ensures each dimension gets to "condition" on others across layers.
    - FlowBatchNorm is applied between MAFLayers to stabilize gradients.
    - The base distribution is a standard multivariate Gaussian: Z ~ N(0, I).

    Parameters
    ----------
    dim : int
        Dimensionality of the input data (D = number of assets, e.g., 3).
    n_layers : int
        Number of MAFLayer steps in the flow.
    hidden_dim : int
        Hidden dimension inside each MADE network.
    n_hidden : int
        Number of hidden layers inside each MADE network.
    context_dim : int, optional
        Dimensionality of the conditioning context h_t.
    use_batch_norm : bool
        Whether to insert FlowBatchNorm between MAFLayers.
    """

    def __init__(
        self,
        dim: int,
        n_layers: int = 5,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        context_dim: Optional[int] = None,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.n_layers = n_layers

        # Build alternating MAFLayer + optional FlowBatchNorm
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                MAFLayer(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    n_hidden=n_hidden,
                    context_dim=context_dim,
                )
            )
            if use_batch_norm and i < n_layers - 1:
                self.layers.append(FlowBatchNorm(dim))

        # Flip indices (dimension ordering) applied between successive MAFLayers
        # This ensures each dimension gets a chance to condition on others
        self.register_buffer(
            "flip_idx",
            torch.arange(dim - 1, -1, -1)
        )

    def log_prob(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the exact log-likelihood of data x under the flow.

        This is the FAST direction for MAF (density estimation at train time).

        log p(x|h) = log p_Z(g(x; h)) + log|det J_g(x; h)|

        where:
            g = the inverse flow (data → noise)
            p_Z = standard Gaussian density
            log|det J| = sum of log-diagonal-Jacobians across all MAFLayers

        Parameters
        ----------
        x : Tensor of shape (batch, D)
        context : Tensor of shape (batch, context_dim), optional

        Returns
        -------
        log_prob : Tensor of shape (batch,)
        """
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        flip = False

        for layer in self.layers:
            if isinstance(layer, MAFLayer):
                if flip:
                    z = z[:, self.flip_idx]
                z, log_det = layer(z, context)
                log_det_total = log_det_total + log_det
                flip = not flip
            else:  # FlowBatchNorm
                z, log_det = layer(z, context)
                log_det_total = log_det_total + log_det

        # Base distribution: standard multivariate Gaussian log-density
        # log p_Z(z) = -D/2 * log(2π) - 0.5 * ||z||^2
        D = x.shape[-1]
        log_pz = (-0.5 * D * torch.log(torch.tensor(2.0 * torch.pi, device=x.device))
                  - 0.5 * (z ** 2).sum(dim=-1))

        return log_pz + log_det_total

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        context: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate samples from the learned distribution via inverse flow.

        z ~ N(0, I)  →  x = f^{-1}(z; h)

        The inverse for MAF is SEQUENTIAL (slow, O(D) × n_layers passes),
        but only needed during inference/Monte Carlo, not training.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        context : Tensor of shape (1, context_dim) or (n_samples, context_dim)
            Conditioning context (broadcast if shape is (1, ...)).

        Returns
        -------
        x_samples : Tensor of shape (n_samples, D)
        """
        if device is None:
            device = next(self.parameters()).device

        # Sample from base distribution
        z = torch.randn(n_samples, self.dim, device=device)

        # Broadcast context if single vector provided
        if context is not None and context.shape[0] == 1:
            context = context.expand(n_samples, -1)

        # Apply inverse flow in REVERSE order
        flip = (sum(1 for l in self.layers if isinstance(l, MAFLayer)) % 2 == 0)
        for layer in reversed(self.layers):
            if isinstance(layer, MAFLayer):
                z = layer.inverse(z, context)
                if flip:
                    z = z[:, self.flip_idx]
                flip = not flip
            else:  # FlowBatchNorm
                z = layer.inverse(z, context)

        return z


if __name__ == "__main__":
    # Sanity check
    D = 3   # 3 assets
    B = 32  # Batch size
    context_dim = 64

    flow = MAFlow(dim=D, n_layers=5, hidden_dim=64, context_dim=context_dim)
    x = torch.randn(B, D)
    h = torch.randn(B, context_dim)

    log_p = flow.log_prob(x, context=h)
    print(f"log_prob shape: {log_p.shape}")       # Expected: (32,)
    print(f"NLL (mean): {-log_p.mean().item():.4f}")

    samples = flow.sample(n_samples=100, context=h[:1])
    print(f"samples shape: {samples.shape}")      # Expected: (100, 3)

    total_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)
    print(f"Total Flow parameters: {total_params:,}")
