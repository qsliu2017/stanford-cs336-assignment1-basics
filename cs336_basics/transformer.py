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
