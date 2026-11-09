"""ChessQueries model head-type contract (query decoder vs. the A1 linear ablation)."""

import pytest
import torch

from camera_chess.chessqueries.models.chessqueries_model import ChessQueriesModel, HeadType

# 98 = 7*14, the smallest DINOv2 grid that still exercises the 8x8 pool.
_X = torch.randint(0, 256, (2, 3, 98, 98)).float() / 255.0


@pytest.mark.parametrize("head_type", list(HeadType))
def test_head_types_emit_board_logits(head_type):
    m = ChessQueriesModel(freeze_encoder=True, head_type=head_type).eval()
    with torch.no_grad():
        out = m(_X)
        labels = m.predict_labels(_X)
    assert out.shape == (2, 64, 13)
    assert labels.shape == (2, 64)
    assert labels.min() >= 0 and labels.max() < 13


def test_linear_head_drops_query_decoder():
    # The A1 ablation must actually remove the queries + transformer decoder,
    # otherwise it is not testing "no query decoder".
    m = ChessQueriesModel(freeze_encoder=True, head_type="linear")
    assert not hasattr(m, "decoder")
    assert not hasattr(m, "queries")
    q = ChessQueriesModel(freeze_encoder=True, head_type="query")
    assert hasattr(q, "decoder") and hasattr(q, "queries")


def test_invalid_head_type_rejected():
    with pytest.raises(ValueError):
        ChessQueriesModel(head_type="bogus")


def test_head_type_accepts_checkpoint_strings():
    """Old checkpoints store head_type as a bare string in hparams; construction
    must normalize it to the enum."""
    m = ChessQueriesModel(pretrained=False, freeze_encoder=True, head_type="linear")
    assert m.head_type is HeadType.LINEAR


def test_forward_with_attention_matches_forward():
    m = ChessQueriesModel(pretrained=False, freeze_encoder=True).eval()
    with torch.no_grad():
        logits, weights = m.forward_with_attention(_X)
        torch.testing.assert_close(logits, m(_X))
    tokens = m.encoder.num_prefix_tokens + (98 // 14) ** 2
    assert weights.shape == (2, 64, tokens)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 64))


def test_forward_with_attention_needs_query_head():
    m = ChessQueriesModel(pretrained=False, freeze_encoder=True, head_type="linear")
    with pytest.raises(ValueError):
        m.forward_with_attention(_X)


CNN = "mobilenetv4_conv_small.e2400_r224_in1k"


def test_cnn_encoder_feeds_the_query_decoder():
    m = ChessQueriesModel(CNN, pretrained=False, freeze_encoder=False).eval()
    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        memory = m.encode(x)
        out = m(x)
    # Stride-16 (8x8) and stride-32 (4x4) maps, projected to one width
    assert memory.shape == (2, 64 + 16, m.level_emb.embedding_dim)
    assert out.shape == (2, 64, 13)


def test_cnn_memory_projection_trains_with_a_frozen_encoder():
    m = ChessQueriesModel(CNN, pretrained=False, freeze_encoder=True)
    m(torch.rand(1, 3, 64, 64)).sum().backward()
    assert m.memory_proj[0].weight.grad is not None
    assert all(p.grad is None for p in m.encoder.parameters())


def test_cnn_encoder_rejects_the_linear_head():
    with pytest.raises(ValueError, match="query head"):
        ChessQueriesModel(CNN, pretrained=False, head_type="linear")
