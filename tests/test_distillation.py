"""Teacher distillation in LitChessQueriesModel."""

import copy

import torch

from camera_chess.chessqueries.train.lit import LitChessQueriesModel

CNN = "mobilenetv4_conv_small.e2400_r224_in1k"


def _lit(teacher=None) -> LitChessQueriesModel:
    return LitChessQueriesModel(encoder_name=CNN, freeze_encoder=False, teacher=teacher)


def test_teacher_stays_out_of_the_state_dict_and_hparams():
    student = _lit()
    lit = _lit(teacher=copy.deepcopy(student.model))
    assert set(lit.state_dict()) == set(student.state_dict())
    assert "teacher" not in lit.hparams
    assert all(not p.requires_grad for p in lit._teacher[0].parameters())


def test_distill_loss_is_zero_for_a_matching_teacher_and_trains_the_student():
    torch.manual_seed(0)
    lit = _lit().eval()  # eval so BatchNorm gives the student and its copy the same outputs
    lit._teacher.append(copy.deepcopy(lit.model).eval())
    x = torch.rand(2, 3, 64, 64)
    assert float(lit._distill_loss(lit(x), x)) < 1e-6
    weight = lit._teacher[0].get_parameter("head.weight")
    weight.data.add_(torch.randn_like(weight))
    loss = lit._distill_loss(lit(x), x)
    assert float(loss) > 0
    loss.backward()
    assert lit.model.get_parameter("head.weight").grad is not None
    assert all(p.grad is None for p in lit._teacher[0].parameters())
