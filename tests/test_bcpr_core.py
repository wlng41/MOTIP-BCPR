"""CPU regression tests for the BCPR module."""

import torch

from models.motip.bcpr import BCPR


def _inputs(feature_dim=32):
    generator = torch.Generator().manual_seed(7)
    q_feat = torch.randn(2, 4, feature_dim, generator=generator)
    h_feat = torch.randn(2, 6, feature_dim, generator=generator)
    valid = torch.tensor([
        [True, True, True, True, True, False],
        [True, True, True, True, False, False],
    ])
    track_id = torch.tensor([
        [0, 0, 1, 1, 2, -1],
        [0, 0, 1, 1, -1, -1],
    ])
    times = torch.tensor([
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0],
    ])
    query_time = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    causal = times[:, None, :] < query_time[:, :, None]
    return q_feat, h_feat, valid, track_id, times, query_time, causal


def test_parameter_count_and_zero_initialisation():
    module = BCPR()
    assert module.parameter_count == 45_378
    bias = module(*_inputs(feature_dim=256))
    assert bias.shape == (2, 4, 6)
    assert torch.count_nonzero(bias) == 0
    assert torch.isfinite(bias).all()


def test_masks_and_row_centering_are_authoritative():
    module = BCPR(feature_dim=32, proj_dim=8, hidden_dim=8, chunk_queries=2)
    with torch.no_grad():
        torch.nn.init.normal_(module.mlp[-1].weight, std=0.2)
        torch.nn.init.normal_(module.mlp[-1].bias, std=0.2)
    inputs = _inputs()
    bias = module(*inputs)
    visible = inputs[-1] & inputs[2].unsqueeze(1)
    assert torch.equal(bias[~visible], torch.zeros_like(bias[~visible]))
    row_sum = (bias * visible).sum(dim=-1)
    assert torch.allclose(row_sum, torch.zeros_like(row_sum), atol=2e-6, rtol=0)


def test_off_and_on_gradient_paths():
    module = BCPR(feature_dim=32, proj_dim=8, hidden_dim=8, chunk_queries=0)
    shared_off = torch.randn(2, 4, 6, requires_grad=True)
    bias = module(*_inputs())
    (shared_off + 0.0 * bias).square().mean().backward()
    off_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    )
    assert off_grad == 0.0
    assert float(shared_off.grad.abs().sum()) > 0.0

    module.zero_grad(set_to_none=True)
    shared_on = shared_off.detach().clone().requires_grad_(True)
    bias = module(*_inputs())
    (shared_on + bias).square().mean().backward()
    first_step_grad = float(module.mlp[-1].weight.grad.abs().sum())
    assert first_step_grad > 0.0
    assert float(shared_on.grad.abs().sum()) > 0.0


def test_weighted_dual_path_objective():
    loss_off = torch.tensor(1.25, requires_grad=True)
    loss_on = torch.tensor(0.75, requires_grad=True)
    objective = 0.5 * loss_off + 0.5 * loss_on
    assert objective.item() == 1.0
    objective.backward()
    assert loss_off.grad.item() == 0.5
    assert loss_on.grad.item() == 0.5
