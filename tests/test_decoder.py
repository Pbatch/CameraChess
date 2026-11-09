"""QueryDecoder matches the nn.TransformerDecoder it replaced: weights, initialisation and outputs."""

import pytest
import torch
from torch import nn

from camera_chess.chessqueries.models.decoder import QueryDecoder

DIM, HEADS, LAYERS = 32, 4, 3


def torch_decoder() -> nn.TransformerDecoder:
    layer = nn.TransformerDecoderLayer(d_model=DIM, nhead=HEADS, dim_feedforward=4 * DIM, batch_first=True)
    return nn.TransformerDecoder(layer, num_layers=LAYERS)


def query_decoder() -> QueryDecoder:
    return QueryDecoder(DIM, HEADS, LAYERS, ff_dim=4 * DIM)


def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    return torch.randn(2, 64, DIM, generator=gen), torch.randn(2, 50, DIM, generator=gen)


def test_same_weights_and_initialisation():
    """Same state dict keys, shapes and, from the same seed, values, so checkpoints and seeded runs carry over"""
    torch.manual_seed(0)
    expected = torch_decoder().state_dict()
    torch.manual_seed(0)
    actual = query_decoder().state_dict()
    assert list(actual) == list(expected)
    for key, tensor in expected.items():
        torch.testing.assert_close(actual[key], tensor, rtol=0, atol=0, msg=key)


@pytest.fixture
def pair() -> tuple[nn.TransformerDecoder, QueryDecoder]:
    reference = torch_decoder().eval()
    decoder = query_decoder().eval()
    decoder.load_state_dict(reference.state_dict(), strict=True)
    return reference, decoder


@torch.no_grad()
def test_matches_transformer_decoder(pair):
    reference, decoder = pair
    queries, memory = inputs()
    h, weights = decoder(queries, memory)
    assert weights is None
    torch.testing.assert_close(h, reference(queries, memory))


@torch.no_grad()
def test_attention_weights_match_multihead_attention(pair):
    reference, decoder = pair
    queries, memory = inputs()
    x = queries
    for layer in reference.layers[:-1]:
        x = layer(x, memory)
    last = reference.layers[-1]
    x = last.norm1(x + last._sa_block(x, None, None))
    _, expected = last.multihead_attn(x, memory, memory, need_weights=True, average_attn_weights=True)
    _, weights = decoder(queries, memory, need_weights=True)
    torch.testing.assert_close(weights, expected)


@torch.no_grad()
def test_unbatched_2d_matches_batched(pair):
    """The LiteRT export runs the layers on (tokens, C) tensors"""
    _, decoder = pair
    queries, memory = inputs()
    batched = decoder(queries, memory, need_weights=True)
    single = decoder(queries[1], memory[1], need_weights=True)
    for b, s in zip(batched, single, strict=True):
        torch.testing.assert_close(s, b[1])


def test_dropout_only_in_training(pair):
    _, decoder = pair
    queries, memory = inputs()
    decoder.train()
    assert not torch.equal(decoder(queries, memory)[0], decoder(queries, memory)[0])
    decoder.eval()
    torch.testing.assert_close(decoder(queries, memory)[0], decoder(queries, memory)[0], rtol=0, atol=0)
