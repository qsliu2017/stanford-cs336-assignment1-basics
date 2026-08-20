import math
from typing import override

import einops
import torch
from jaxtyping import Float, Int
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


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        i_s = torch.range(0, max_seq_len - 1, dtype=torch.int, device=device).reshape([max_seq_len, 1])
        assert d_k % 2 == 0
        n_pair = d_k // 2
        # -(2k-2)/d for k \in {1,...,d/2}
        inversed_frequences = torch.range(0, n_pair - 1, device=device).mul_(2).neg_().div_(d_k)

        # theta_{i,k}
        thetas = i_s * (theta**inversed_frequences)
        assert thetas.shape == torch.Size([max_seq_len, n_pair]), (
            f"expect [{max_seq_len}, {n_pair}], got {thetas.shape}"
        )
        cos = torch.cos(thetas)
        sin = torch.sin(thetas)
        cos = einops.rearrange(einops.repeat(cos, "i k -> i k repeat", repeat=2), "i k repeat -> i (k repeat)")
        sin = einops.rearrange(einops.repeat(sin, "i k -> i k repeat", repeat=2), "i k repeat -> i (k repeat)")

        assert cos.shape == torch.Size([max_seq_len, d_k]), f"{cos.shape}"
        assert cos[-1][0] == cos[-1][1]
        self.cos: Float[Tensor, " max_seq_len d_k"] = cos
        self.sin: Float[Tensor, " max_seq_len d_k"] = sin

        k_s = torch.range(0, d_k - 1, dtype=torch.int, device=device)
        self.in_pair_even_index: Int[Tensor, " d_k"] = k_s // 2 * 2
        self.in_pair_odd_index: Int[Tensor, " d_k"] = k_s // 2 * 2 + 1

        self.even_mask: Int[Tensor, " d_k"] = k_s % 2 == 0
        self.odd_mask: Int[Tensor, " d_k"] = k_s % 2 == 1

    @override
    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ) -> Float[Tensor, " ... sequence_length d_k"]:
        in_shape = x.shape
        x = einops.rearrange(x, " ... sequence_length d_k ->  (... sequence_length) d_k")
        token_positions = einops.repeat(
            token_positions,
            # repeat along the sequence dimension.
            " ... sequence_length -> (... repeat sequence_length)",
            repeat=x.shape[0] // token_positions.numel(),
        )

        x_even = x.index_select(-1, self.in_pair_even_index)
        x_odd = x.index_select(-1, self.in_pair_odd_index)
        cos = self.cos.index_select(0, token_positions)
        sin = self.sin.index_select(0, token_positions)

        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos
        y = einops.einsum(y_even, self.even_mask, "tokens k, k -> tokens k") + einops.einsum(
            y_odd, self.odd_mask, "tokens k, k -> tokens k"
        )
        return y.reshape(in_shape)
