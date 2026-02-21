"""Core neural-network building blocks for a GPT-2-style Transformer."""

import math
import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    """Linear transformation y = Wx (no bias), following modern LLM convention.

    Parameters are initialized with a truncated normal distribution:
        W ~ N(0, σ²)  truncated at [-3σ, 3σ],  where σ² = 2 / (d_in + d_out).

    We store W as shape (d_out, d_in) and use einops.einsum for the forward
    pass, which avoids the need to manually transpose — the Einstein summation
    notation makes the contraction dimension explicit regardless of memory layout.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # W stored as (d_out, d_in) — same convention as nn.Linear
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # Initialize: truncated normal N(0, σ²), σ² = 2/(d_in+d_out), truncated at [-3σ, 3σ]
        self._reset_parameters()

    def _reset_parameters(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation using einsum.

        Math (column-vector convention):  y = W x
        Code (einsum, layout-agnostic):   y_...j = sum_i(x_...i * W_ji)

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings : int,
        embedding_dim : int,
        device : torch.device | None = None,
        dtype : torch.dtype | None = None,
    ):
        """
        num_enbeddings: size of vocabulary
        embeddings_dim: dimension of the embedding model . e.g. d_model
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3, b=3)

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Look up the embedding vectors for the given token IDs.

        Arg s:
            token_ids: Integer tensor of shape (...) with token indices.

        Returns:
            Tensor of shape (..., embedding_dim) with the corresponding embeddings.
        """
        # Advanced indexing: weight[token_ids] gathers rows from the embedding matrix.
        # token_ids shape: (...)  →  output shape: (..., embedding_dim)
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    RMSNorm(a_i) = a_i / RMS(a) * g_i

    where RMS(a) = sqrt(1/d_model * sum(a_i^2) + eps)
    and g_i is a learnable gain parameter initialized to 1.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Learnable gain parameter, initialized to 1
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to the input.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Normalized tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # RMS(a) = sqrt(mean(a^2) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and apply gain
        result = (x / rms) * self.weight

        return result.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU / Swish activation: SiLU(x) = x * σ(x).

    Uses torch.sigmoid for numerical stability as recommended.
    """
    return x * torch.sigmoid(x)


def compute_d_ff(d_model: int) -> int:
    """Compute d_ff ≈ 8/3 * d_model, rounded up to the nearest multiple of 64."""
    d_ff = int(8/3 * d_model)
    return ((d_ff + 63)//64) * 64


class SwiGLU(nn.Module):
    """SwiGLU position-wise feed-forward network.

    FFN(x) = W₂(SiLU(W₁x) ⊙ W₃x)

    W₁: (d_ff, d_model)  — gate projection
    W₃: (d_ff, d_model)  — value projection
    W₂: (d_model, d_ff)  — down projection
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = compute_d_ff(d_model)
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # gate
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # down
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU: W₂(SiLU(W₁x) ⊙ W₃x).

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Output tensor of shape (..., d_model).
        """
        gate = silu(self.w1(x))   # SiLU(W₁x):  (..., d_ff)
        value = self.w3(x)        # W₃x:        (..., d_ff)
        return self.w2(gate * value)  # W₂(gate ⊙ value): (..., d_model)
