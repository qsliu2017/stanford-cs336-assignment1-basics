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
