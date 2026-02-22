"""Core neural-network building blocks for a GPT-2-style Transformer."""

import math
import torch
import torch.nn as nn
from einops import einsum, rearrange


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


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE) — Su et al., 2021.

    Precomputes cos(θ_{i,k}) and sin(θ_{i,k}) for all positions i and
    frequency indices k, then applies pairwise 2D rotation to the
    input query/key vectors.

    θ_{i,k} = i / Θ^{(2k-2)/d}   for k ∈ {1, ..., d/2}
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.d_k = d_k

        # Frequency for each pair index k: 1 / Θ^{(2k) / d_k}  for k = 0..d_k/2-1
        # Using (2k) / d_k as exponent (0-indexed: k=0 gives Θ^0, k=1 gives Θ^{2/d}, ...)
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        # freqs shape: (d_k // 2,)

        # positions shape: (max_seq_len,)
        positions = torch.arange(max_seq_len, device=device).float()

        # angles shape: (max_seq_len, d_k // 2)
        angles = einsum(positions, freqs, "i, j -> i j")

        # Precompute cos and sin, register as non-persistent buffers (not saved in state_dict)
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RoPE rotation to query or key tensor.

        Args:
            x: Tensor of shape (..., seq_len, d_k).
            token_positions: Integer tensor of shape (..., seq_len)
                specifying the absolute position of each token.

        Returns:
            Rotated tensor of the same shape (..., seq_len, d_k).
        """
        # Gather cos/sin for the given positions
        # cos_cached: (max_seq_len, d_k//2)  →  index with token_positions
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k//2)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k//2)

        # Reshape x into pairs: (..., seq_len, d_k) → (..., seq_len, d_k//2, 2)
        x_pairs = rearrange(x, "... (pairs two) -> ... pairs two", two=2)

        # x_even = x[..., 0::2], x_odd = x[..., 1::2]
        x_even = x_pairs[..., 0]  # (..., seq_len, d_k//2)
        x_odd = x_pairs[..., 1]   # (..., seq_len, d_k//2)

        # Apply 2D rotation to each pair:
        #   [cos  -sin] [x_even]   [cos*x_even - sin*x_odd]
        #   [sin   cos] [x_odd ] = [sin*x_even + cos*x_odd]
        rotated_even = cos * x_even - sin * x_odd
        rotated_odd = sin * x_even + cos * x_odd

        # Interleave back: stack on last dim and flatten
        # (..., seq_len, d_k//2, 2) → (..., seq_len, d_k)
        # When passing a list to rearrange, the new dimension 'two' is at the front (index 0)
        rotated = rearrange([rotated_even, rotated_odd], "two ... pairs -> ... (pairs two)", two=2)
        return rotated

def softmax(
    x: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    max_o = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_o)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Args:
        Q: Query tensor of shape (..., n, d_k).
        K: Key tensor of shape (..., m, d_k).
        V: Value tensor of shape (..., m, d_v).
        mask: Optional boolean mask of shape (..., n, m).
              True = attend, False = do not attend.

    Returns:
        Output tensor of shape (..., n, d_v).
    """
    d_k = Q.shape[-1]

    # Q K^T / sqrt(d_k)  →  (..., n, m)
    scores = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / math.sqrt(d_k)

    # Apply mask: set False positions to -inf so softmax gives them 0 weight
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # Softmax over the key dimension (last dim)
    weights = softmax(scores, dim=-1)

    # Weighted sum of values: (..., n, m) @ (..., m, d_v) → (..., n, d_v)
    return einsum(weights, V, "... n m, ... m d_v -> ... n d_v")

class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention.

    MultiHeadSelfAttention(x) = W_O · MultiHead(W_Q·x, W_K·x, W_V·x)

    where each head independently computes scaled dot-product attention
    with a causal mask (each position can only attend to earlier positions).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # d_k = d_v = d_model / h

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply causal multi-head self-attention.

        Args:
            x: Input tensor of shape (..., seq_len, d_model).

        Returns:
            Output tensor of shape (..., seq_len, d_model).
        """
        seq_len = x.shape[-2]

        # Project Q, K, V: (..., seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into heads: (..., seq_len, d_model) → (..., num_heads, seq_len, d_k)
        Q = rearrange(Q, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        K = rearrange(K, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        V = rearrange(V, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        
        # Build causal mask: (seq_len, seq_len), True = allowed to attend
        # Position i can attend to positions j <= i
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        ).logical_not()  # upper triangle (excluding diagonal) → flip to lower triangle + diagonal

        # Apply scaled dot-product attention per head (head dim is a batch dim)
        # (..., num_heads, seq_len, d_k) → (..., num_heads, seq_len, d_k)
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # Merge heads: (..., num_heads, seq_len, d_k) → (..., seq_len, d_model)
        attn_output = rearrange(attn_output, "... h seq d_k -> ... seq (h d_k)")

        # Output projection: (..., seq_len, d_model) → (..., seq_len, d_model)
        return self.output_proj(attn_output)

class MultiHeadSelfAttentionWithRope(nn.Module):
    """Causal multi-head self-attention with Rotary Positional Embeddings (RoPE)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply causal multi-head self-attention with RoPE."""
        seq_len = x.shape[-2]

        # Project Q, K, V: (..., seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into heads: (..., seq_len, d_model) -> (..., num_heads, seq_len, d_k)
        Q = rearrange(Q, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        K = rearrange(K, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        V = rearrange(V, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)

        # Prepare token positions and apply RoPE to Q and K (not V).
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        if token_positions.device != x.device:
            token_positions = token_positions.to(x.device)

        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        # Build causal mask: (seq_len, seq_len), True = allowed to attend
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        ).logical_not()

        # Apply scaled dot-product attention per head
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # Merge heads: (..., num_heads, seq_len, d_k) -> (..., seq_len, d_model)
        attn_output = rearrange(attn_output, "... h seq d_k -> ... seq (h d_k)")

        # Output projection: (..., seq_len, d_model)
        return self.output_proj(attn_output)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttentionWithRope(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)
