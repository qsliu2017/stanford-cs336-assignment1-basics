import math
from typing import override

import einops
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        weights: Float[Tensor, " out in"] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if weights is None:
            w = torch.zeros([out_features, in_features], device=device, dtype=dtype)
            std = math.sqrt(2.0 / float(in_features + out_features))
            weights = torch.nn.init.trunc_normal_(w, mean=0, std=std, a=3 * std, b=-3 * std)
        self.w: torch.nn.Parameter = torch.nn.Parameter(weights)

    @override
    def forward(self, x: Float[Tensor, " seq in"]) -> Float[Tensor, " seq out"]:
        return einops.einsum(
            self.w,
            x,
            "out in, ... in -> ... out",
        )


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        weights: Float[Tensor, " vocab_size d_model"] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if weights is None:
            w = torch.zeros([num_embeddings, embedding_dim], device=device, dtype=dtype)
            weights = torch.nn.init.trunc_normal_(w, mean=0, std=1, a=3, b=-3)
        self.w: torch.nn.Parameter = torch.nn.Parameter(weights)

    @override
    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        shape = token_ids.shape
        indices = token_ids.flatten()

        embeddings = torch.index_select(self.w, 0, indices)
        d_model = self.w.shape[-1]
        return embeddings.reshape([*shape, d_model])


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        *,
        data: Float[torch.Tensor, " d_model"] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps: float = eps
        if data is None:
            data = torch.ones([d_model], device=device, dtype=dtype)
        self.g: Float[torch.Tensor, " d_model"] = data

    @override
    def forward(self, x: Float[torch.Tensor, " ... d_model"]) -> Float[torch.Tensor, " ... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = einops.reduce(torch.pow(x, 2), "... seq d_model -> ... seq", "mean").add(self.eps).pow(-0.5)
        a = einops.einsum(rms, x, "... seq, ... seq d_model -> ... seq d_model")
        norm = einops.einsum(a, self.g, "... seq d_model, d_model -> ... seq d_model")

        return norm.to(in_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        w1_weight: Float[torch.Tensor, " d_ff d_model"] | None = None,
        w2_weight: Float[torch.Tensor, " d_model d_ff"] | None = None,
        w3_weight: Float[torch.Tensor, " d_ff d_model"] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if w1_weight is None:
            w1_weight = torch.rand([d_ff, d_model])
        if w2_weight is None:
            w2_weight = torch.rand([d_model, d_ff])
        if w3_weight is None:
            w3_weight = torch.rand([d_ff, d_model])
        self.d_model: int = d_model
        self.d_ff: int = d_ff

        self.w1: Float[torch.Tensor, " d_ff d_model"] = w1_weight
        self.w2: Float[torch.Tensor, " d_model d_ff"] = w2_weight
        self.w3: Float[torch.Tensor, " d_ff d_model"] = w3_weight

    @override
    def forward(
        self,
        x: Float[torch.Tensor, " ... d_model"],
    ) -> Float[torch.Tensor, " ... d_model"]:
        w1x = einops.einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = w1x * torch.sigmoid(w1x)
        w3x = einops.einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        swiglu = einops.einsum(self.w2, silu * w3x, "d_model d_ff, ... d_ff -> ... d_model")
        return swiglu


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        i_s = torch.arange(0, max_seq_len, dtype=torch.int, device=device).reshape([max_seq_len, 1])
        assert d_k % 2 == 0
        n_pair = d_k // 2
        # -(2k-2)/d for k \in {1,...,d/2}
        inversed_frequences = torch.arange(0, n_pair, device=device, dtype=torch.float32).mul_(2).neg_().div_(d_k)

        # theta_{i,k}
        thetas = i_s * (theta**inversed_frequences)
        self.register_buffer("freqs_cis", torch.polar(torch.ones_like(thetas), thetas))

    @override
    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ) -> Float[Tensor, " ... sequence_length d_k"]:
        in_shape = x.shape
        in_dtype = x.dtype
        x = einops.rearrange(x, " ... (d_k pair) ->  (...) d_k pair", pair=2)
        x = x.type(torch.float32)
        x = torch.view_as_complex(x)

        token_positions = einops.repeat(
            token_positions,
            " ... sequence_length -> (... repeat sequence_length)",
            repeat=x.shape[0] // token_positions.numel(),
        )
        freqs_cis = self.get_buffer("freqs_cis").index_select(0, token_positions)
        y = x * freqs_cis
        return torch.view_as_real(y).reshape(in_shape).type(in_dtype)


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    max_, _ = torch.max(in_features, dim, keepdim=True)
    exp_ = torch.exp(in_features - max_)
    sum_ = torch.sum(exp_, dim, keepdim=True)
    softmax = exp_ / sum_

    return softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    device = Q.device
    if mask is None:
        queries = Q.shape[-2]
        keys = K.shape[-2]
        q_s = torch.arange(0, queries, dtype=torch.int, device=device).reshape([queries, 1])
        k_s = torch.arange(0, keys, dtype=torch.int, device=device).reshape([1, keys])
        mask = q_s >= k_s
        assert mask.shape == (Q.shape[-2], K.shape[-2])

    qk = einops.einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    addon = torch.zeros_like(qk)
    addon = addon.masked_fill_(mask.logical_not_(), -torch.inf)
    masked_qk: Float[Tensor, "... queries keys"] = (qk + addon).div(math.sqrt(d_k))
    softmaxed = softmax(masked_qk, -1)
    attn = einops.einsum(softmaxed, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return attn


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[Tensor, " d_model d_model"] | None = None,
        k_proj_weight: Float[Tensor, " d_model d_model"] | None = None,
        v_proj_weight: Float[Tensor, " d_model d_model"] | None = None,
        o_proj_weight: Float[Tensor, " d_model d_model"] | None = None,
        theta: float | None = None,
        max_seq_len: int | None = None,
    ):
        super().__init__()
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        if q_proj_weight is None:
            q_proj_weight = torch.rand([d_model, d_model])
        if k_proj_weight is None:
            k_proj_weight = torch.rand([d_model, d_model])
        if v_proj_weight is None:
            v_proj_weight = torch.rand([d_model, d_model])
        if o_proj_weight is None:
            o_proj_weight = torch.rand([d_model, d_model])
        self.wqkv: torch.nn.Parameter = torch.nn.Parameter(torch.concat([q_proj_weight, k_proj_weight, v_proj_weight]))
        self.wo: torch.nn.Parameter = torch.nn.Parameter(o_proj_weight)

        self.rope: RotaryPositionalEmbedding | None = None
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=q_proj_weight.device)

    @override
    def forward(
        self,
        in_features: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        seq_len = in_features.shape[-2]
        q, k, v = einops.einsum(
            self.wqkv, in_features, "triple_d_model d_model, ... seq d_model -> ... seq triple_d_model"
        ).tensor_split(3, -1)
        assert q.shape[:-2] == in_features.shape[:-2]
        q_heads = einops.rearrange(q, "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads=self.num_heads)
        k_heads = einops.rearrange(k, "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads=self.num_heads)
        v_heads = einops.rearrange(v, "... seq (num_heads d_v) -> ... num_heads seq d_v", num_heads=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(in_features.shape[-2], dtype=torch.int, device=in_features.device)
            qk_heads = torch.concat([q_heads, k_heads])
            q_heads, k_heads = self.rope.forward(qk_heads, token_positions).tensor_split(2, 0)

        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=in_features.device).tril_()
        attn_heads = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)
        attn = einops.rearrange(attn_heads, "... num_heads seq d_v -> ... seq (num_heads d_v)")

        o = einops.einsum(self.wo, attn, "d_model hdv, ... seq hdv -> ... seq d_model")
        return o
