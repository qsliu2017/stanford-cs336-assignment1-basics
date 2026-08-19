import math
from typing import override

import einops
import torch
from jaxtyping import Float, Int


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        weights: Float[torch.Tensor, " out in"] | None = None,
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
    def forward(self, x: Float[torch.Tensor, " seq in"]) -> Float[torch.Tensor, " seq out"]:
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
        weights: Float[torch.Tensor, " vocab_size d_model"] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if weights is None:
            w = torch.zeros([num_embeddings, embedding_dim], device=device, dtype=dtype)
            weights = torch.nn.init.trunc_normal_(w, mean=0, std=1, a=3, b=-3)
        self.w: torch.nn.Parameter = torch.nn.Parameter(weights)

    @override
    def forward(self, token_ids: Int[torch.Tensor, " ..."]) -> Float[torch.Tensor, " ... d_model"]:
        shape = token_ids.shape
        indices = token_ids.flatten()

        embeddings = torch.index_select(self.w, 0, indices)
        d_model = self.w.shape[-1]
        return embeddings.reshape([*shape, d_model])
