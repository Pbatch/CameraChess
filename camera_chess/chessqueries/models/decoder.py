"""The square-query decoder: nn.TransformerDecoder's post-norm layers, written out.

Parameter names, initialisation and maths match ``nn.TransformerDecoder`` of
``nn.TransformerDecoderLayer(batch_first=True)`` (ReLU, dropout 0.1, post-norm), so checkpoints
saved from that load unchanged, and a seed gives the same initial weights.

Written out so the same code runs batched, (B, tokens, C), for training and on 2D (tokens, C)
tensors for the LiteRT export (scripts/export_litert.py), whose WebGPU delegate mislays out 3D
linear outputs and rejects tensors above 3D on the way into attention. It also returns the
last layer's cross-attention weights, which the browser model turns into the board orientation.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    dropout_p: float = 0.0,
    need_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Multi-head attention on (..., N, C) queries and (..., M, C) keys and values, batched or not.

    Heads are split on one extra dimension, so 2D inputs never go above 3D: (heads, N, d).
    With need_weights, also returns the (..., N, M) attention weights averaged over heads,
    before dropout."""
    d = q.shape[-1] // heads
    qh = (q * d**-0.5).unflatten(-1, (heads, d)).transpose(-3, -2)  # (..., H, N, d)
    kh = k.unflatten(-1, (heads, d)).movedim(-3, -1)  # (..., H, d, M)
    vh = v.unflatten(-1, (heads, d)).transpose(-3, -2)  # (..., H, M, d)
    probs = torch.softmax(qh @ kh, dim=-1)
    weights = probs.mean(-3) if need_weights else None
    if dropout_p > 0.0:
        probs = F.dropout(probs, dropout_p)
    return (probs @ vh).transpose(-3, -2).flatten(-2), weights


class MultiheadAttention(nn.Module):
    """nn.MultiheadAttention's parameters (one packed q/k/v projection), for attention()."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.embed_dim, self.num_heads, self.dropout = dim, heads, dropout
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))
        self.out_proj = nn.Linear(dim, dim)
        # nn.MultiheadAttention's initialisation, in its order (out_proj's draws come first)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.zeros_(self.out_proj.bias)

    def projection(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The weight and bias of the q (0), k (1) or v (2) projection"""
        c = self.embed_dim
        return self.in_proj_weight[i * c : (i + 1) * c], self.in_proj_bias[i * c : (i + 1) * c]

    def forward(
        self, query: torch.Tensor, memory: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        out, weights = attention(
            F.linear(query, *self.projection(0)),
            F.linear(memory, *self.projection(1)),
            F.linear(memory, *self.projection(2)),
            self.num_heads,
            self.dropout if self.training else 0.0,
            need_weights,
        )
        return self.out_proj(out), weights


class QueryDecoderLayer(nn.Module):
    """nn.TransformerDecoderLayer with norm_first=False: self-attention, cross-attention, then a
    ReLU feed-forward, each added to the residual and followed by a LayerNorm."""

    def __init__(self, dim: int, heads: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(dim, heads, dropout)
        self.multihead_attn = MultiheadAttention(dim, heads, dropout)
        self.linear1 = nn.Linear(dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, dim)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.dropout1, self.dropout2, self.dropout3 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)

    def self_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout1(self.self_attn(x, x)[0])

    def cross_block(
        self, x: torch.Tensor, memory: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        out, weights = self.multihead_attn(x, memory, need_weights)
        return self.dropout2(out), weights

    def ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout3(self.linear2(self.dropout(F.relu(self.linear1(x)))))

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.norm1(x + self.self_block(x))
        attended, weights = self.cross_block(x, memory, need_weights)
        x = self.norm2(x + attended)
        return self.norm3(x + self.ff_block(x)), weights


class QueryDecoder(nn.Module):
    """A stack of QueryDecoderLayers, all starting from the same weights, as nn.TransformerDecoder's."""

    def __init__(self, dim: int, heads: int, num_layers: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        layer = QueryDecoderLayer(dim, heads, ff_dim, dropout)
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))

    def forward(
        self, queries: torch.Tensor, memory: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The decoded queries and, with need_weights, the last layer's cross-attention weights
        averaged over heads, (..., queries, memory tokens)"""
        h, weights = queries, None
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            h, weights = layer(h, memory, need_weights and i == last)
        return h, weights
