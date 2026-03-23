"""
maf.py
------
Masked Autoregressive Flow (MAF) for conditional density estimation.

Implements a stack of invertible MADE-based affine layers. The autoregressive
structure ensures a lower-triangular Jacobian, making log|det J| computable
in O(D) rather than O(D^3).

References:
    - Papamakarios et al. (2017): "Masked Autoregressive Flow for Density Estimation"
    - Germain et al. (2015): "MADE: Masked Autoencoder for Distribution Estimation"
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    """
    Linear layer with a binary mask applied to the weight matrix.

    Used to enforce the autoregressive property in MADE: output unit i
    only receives information from input units j where mask[i, j] = 1.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask: torch.Tensor) -> None:
        """Set the autoregressive mask."""
        self.mask.data.copy_(mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation (MADE).

    A feedforward network with masked weights enforcing the autoregressive
    ordering: output for dimension i depends only on inputs 1, ..., i-1.
    An external context vector h_t can be injected by concatenation.

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
        Activation function: 'relu', 'tanh', or 'elu'.
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

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.context_dim = context_dim

        self.context_proj: Optional[nn.Linear] = None
        effective_input = input_dim
        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, context_dim)
            effective_input = input_dim + context_dim

        layer_sizes = [effective_input] + [hidden_dim] * n_hidden + [input_dim * 2]
        self.layers = nn.ModuleList([
            MaskedLinear(in_sz, out_sz)
            for in_sz, out_sz in zip(layer_sizes[:-1], layer_sizes[1:])
        ])

        self.activation = {"relu": F.relu, "tanh": torch.tanh, "elu": F.elu}[activation]
        self._setup_masks(effective_input, hidden_dim, input_dim)

    def _setup_masks(
        self,
        effective_input: int,
        hidden_size: int,
        output_size: int,
    ) -> None:
        """
        Construct autoregressive masks for each MADE layer.

        Each unit is assigned a degree m(k) in [1, D-1]. A connection from
        unit j to unit i is allowed only if m(j) <= m(i) for hidden layers,
        and m(j) < m(i) for the output layer (strict lower-triangular Jacobian).
        Context units are assigned degree 0 so they are visible to all units.
        """
        input_dim = self.input_dim
        context_dim = self.context_dim if self.context_dim is not None else 0

        if context_dim > 0:
            m_input = np.concatenate([
                np.zeros(context_dim, dtype=int),
                np.arange(1, input_dim + 1, dtype=int),
            ])
        else:
            m_input = np.arange(1, input_dim + 1, dtype=int)

        rng = np.random.default_rng(seed=42)
        m_hidden = [
            rng.integers(low=1, high=input_dim, size=self.hidden_dim, endpoint=False)
            for _ in range(self.n_hidden)
        ]

        # alpha and mu share the same autoregressive order (degrees 0..D-1 each)
        m_output = np.concatenate([np.arange(0, input_dim), np.arange(0, input_dim)])

        all_m = [m_input] + m_hidden + [m_output]
        for i, layer in enumerate(self.layers):
            mask = torch.tensor(
                (all_m[i + 1][:, None] >= all_m[i][None, :]).astype(np.float32)
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
        context : Tensor of shape (batch, context_dim), optional

        Returns
        -------
        alpha : Tensor of shape (batch, D) — log-scale parameters.
        mu : Tensor of shape (batch, D) — shift parameters.
        """
        if context is not None and self.context_proj is not None:
            h = torch.cat([self.context_proj(context), x], dim=-1)
        else:
            h = x

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)

        alpha, mu = h.chunk(2, dim=-1)
        alpha = 3.0 * torch.tanh(alpha / 3.0)  # bound alpha for numerical stability
        return alpha, mu


class MAFLayer(nn.Module):
    """
    Single step of the Masked Autoregressive Flow.

    Affine transformation:
        Forward (data → noise):  z_i = (x_i - mu_i(x_{<i}; h)) * exp(-alpha_i(x_{<i}; h))
        Inverse (noise → data):  x_i = z_i * exp(alpha_i) + mu_i

    Log Jacobian determinant:
        log|det J| = -sum_i(alpha_i)   [triangular, O(D)]

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
        Map data x → noise z (training direction, used for NLL computation).

        Parameters
        ----------
        x : Tensor of shape (batch, D)
        context : Tensor of shape (batch, context_dim), optional

        Returns
        -------
        z : Tensor of shape (batch, D)
        log_det_J : Tensor of shape (batch,)
        """
        alpha, mu = self.made(x, context)
        z = (x - mu) * torch.exp(-alpha)
        log_det_J = -alpha.sum(dim=-1)
        return z, log_det_J

    @torch.no_grad()
    def inverse(
        self,
        z: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Map noise z → data x (sampling direction).

        Sequential by nature: x_i must be computed before x_{i+1},
        requiring D forward passes through MADE.

        Parameters
        ----------
        z : Tensor of shape (batch, D)
        context : Tensor of shape (batch, context_dim), optional

        Returns
        -------
        x : Tensor of shape (batch, D)
        """
        x = torch.zeros_like(z)
        for i in range(z.shape[-1]):
            alpha, mu = self.made(x, context)
            x[:, i] = z[:, i] * torch.exp(alpha[:, i]) + mu[:, i]
        return x


class FlowBatchNorm(nn.Module):
    """
    Invertible batch normalization for normalizing flows.

    Uses running statistics to keep forward and inverse passes consistent
    at inference time. Stabilizes training between flow layers.

    Parameters
    ----------
    dim : int
        Feature dimensionality.
    eps : float
        Numerical stability constant.
    momentum : float
        Running statistics momentum.
    """

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize x → z and return log_det_J."""
        if self.training:
            mean = x.mean(0)
            var = x.var(0, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            mean, var = self.running_mean, self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        z = x_hat * torch.exp(self.gamma) + self.beta

        log_det_J = (self.gamma - 0.5 * torch.log(var + self.eps)).sum()
        return z, log_det_J.expand(x.shape[0])

    @torch.no_grad()
    def inverse(
        self,
        z: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Invert batch normalization: z → x."""
        x_hat = (z - self.beta) * torch.exp(-self.gamma)
        return x_hat * torch.sqrt(self.running_var + self.eps) + self.running_mean


class MAFlow(nn.Module):
    """
    Full Masked Autoregressive Flow: a stack of MAFLayer steps.

    Dimension ordering is reversed between successive MAFLayers so that each
    dimension conditions on different subsets across layers. FlowBatchNorm
    is optionally inserted between layers to stabilize gradients.

    Parameters
    ----------
    dim : int
        Dimensionality of the input data (D = number of assets).
    n_layers : int
        Number of MAFLayer steps.
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

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(MAFLayer(dim=dim, hidden_dim=hidden_dim,
                                        n_hidden=n_hidden, context_dim=context_dim))
            if use_batch_norm and i < n_layers - 1:
                self.layers.append(FlowBatchNorm(dim))

        self.register_buffer("flip_idx", torch.arange(dim - 1, -1, -1))

    def log_prob(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute exact log-likelihood of x under the flow.

        log p(x|h) = log p_Z(g(x; h)) + log|det J_g(x; h)|

        where g is the data→noise direction and p_Z = N(0, I).

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
            else:
                z, log_det = layer(z, context)
                log_det_total = log_det_total + log_det

        D = x.shape[-1]
        log_pz = (
            -0.5 * D * torch.log(torch.tensor(2.0 * torch.pi, device=x.device))
            - 0.5 * (z ** 2).sum(dim=-1)
        )
        return log_pz + log_det_total

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        context: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate samples via the inverse flow: z ~ N(0, I) → x = f^{-1}(z; h).

        The inverse is sequential — O(D) × n_layers MADE passes. Only used
        at inference time for Monte Carlo risk estimation.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        context : Tensor of shape (1, context_dim) or (n_samples, context_dim), optional

        Returns
        -------
        x_samples : Tensor of shape (n_samples, D)
        """
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(n_samples, self.dim, device=device)

        if context is not None and context.shape[0] == 1:
            context = context.expand(n_samples, -1)

        flip = (sum(1 for l in self.layers if isinstance(l, MAFLayer)) % 2 == 0)
        for layer in reversed(self.layers):
            if isinstance(layer, MAFLayer):
                z = layer.inverse(z, context)
                if flip:
                    z = z[:, self.flip_idx]
                flip = not flip
            else:
                z = layer.inverse(z, context)

        return z


if __name__ == "__main__":
    D, B, context_dim = 3, 32, 64
    flow = MAFlow(dim=D, n_layers=5, hidden_dim=64, context_dim=context_dim)
    x = torch.randn(B, D)
    h = torch.randn(B, context_dim)

    log_p = flow.log_prob(x, context=h)
    print(f"log_prob shape: {log_p.shape}")
    print(f"NLL (mean):     {-log_p.mean().item():.4f}")

    samples = flow.sample(n_samples=100, context=h[:1])
    print(f"samples shape:  {samples.shape}")
    print(f"Parameters:     {sum(p.numel() for p in flow.parameters() if p.requires_grad):,}")