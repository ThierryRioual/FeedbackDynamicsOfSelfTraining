import math

import pytest
import torch

from src.objectives import HardSelection, SmoothSelection


KAPPA = math.log(4.0)


def test_default_mode_restores_original_smooth_forward_values():
    epsilon = 0.20
    x = torch.tensor(
        [
            -KAPPA - epsilon,
            -KAPPA,
            -KAPPA + epsilon,
            0.0,
            KAPPA - epsilon,
            KAPPA,
            KAPPA + epsilon,
        ],
        dtype=torch.float64,
    )

    mask = SmoothSelection(epsilon)(
        x, pos_margin=KAPPA, neg_margin=-KAPPA
    )

    torch.testing.assert_close(
        mask,
        torch.tensor(
            [1.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0],
            dtype=x.dtype,
        ),
    )


def test_hard_forward_changes_values_but_not_smooth_derivative():
    epsilon = 0.04
    values = torch.tensor([-KAPPA, KAPPA], dtype=torch.float64)
    soft_input = values.clone().requires_grad_()
    hard_input = values.clone().requires_grad_()
    soft_selector = SmoothSelection(epsilon, False)
    hard_selector = SmoothSelection(epsilon, True)

    soft_mask = soft_selector(
        soft_input, pos_margin=KAPPA, neg_margin=-KAPPA
    )
    hard_mask = hard_selector(
        hard_input, pos_margin=KAPPA, neg_margin=-KAPPA
    )
    soft_derivative = torch.autograd.grad(soft_mask.sum(), soft_input)[0]
    hard_derivative = torch.autograd.grad(hard_mask.sum(), hard_input)[0]

    torch.testing.assert_close(soft_mask, torch.full_like(soft_mask, 0.5))
    assert torch.equal(
        hard_mask.detach(),
        HardSelection()(hard_input, pos_margin=KAPPA, neg_margin=-KAPPA),
    )
    assert torch.equal(hard_mask.detach(), torch.ones_like(hard_mask))
    expected_derivative = torch.tensor(
        [-1.0 / epsilon, 1.0 / epsilon], dtype=values.dtype
    )
    torch.testing.assert_close(soft_derivative, expected_derivative)
    torch.testing.assert_close(hard_derivative, expected_derivative)


@pytest.mark.parametrize("hard_forward", [False, True])
def test_smooth_derivative_is_flat_at_and_outside_epsilon_endpoints(
    hard_forward,
):
    epsilon = 0.05
    x = torch.tensor(
        [
            -KAPPA - 2.0 * epsilon,
            -KAPPA - epsilon,
            -KAPPA + epsilon,
            KAPPA - epsilon,
            KAPPA + epsilon,
            KAPPA + 2.0 * epsilon,
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    selector = SmoothSelection(epsilon, hard_forward)

    mask = selector(x, pos_margin=KAPPA, neg_margin=-KAPPA)
    derivative = torch.autograd.grad(mask.sum(), x)[0]

    torch.testing.assert_close(derivative, torch.zeros_like(derivative))


def test_epsilon_controls_both_modes_with_the_same_smooth_surrogate():
    epsilon = 0.03
    values = torch.tensor(
        [KAPPA - 0.015, KAPPA + 0.015],
        dtype=torch.float64,
    )
    soft_input = values.clone().requires_grad_()
    hard_input = values.clone().requires_grad_()
    soft_selector = SmoothSelection(epsilon)
    hard_selector = SmoothSelection(epsilon, True)

    soft_mask = soft_selector(
        soft_input, pos_margin=KAPPA, neg_margin=-KAPPA
    )
    hard_mask = hard_selector(
        hard_input, pos_margin=KAPPA, neg_margin=-KAPPA
    )
    soft_derivative = torch.autograd.grad(soft_mask.sum(), soft_input)[0]
    hard_derivative = torch.autograd.grad(hard_mask.sum(), hard_input)[0]

    assert torch.equal(
        hard_mask.detach(), torch.tensor([0.0, 1.0], dtype=values.dtype)
    )
    assert torch.all((soft_mask > 0.0) & (soft_mask < 1.0))
    torch.testing.assert_close(soft_mask.sum(), soft_mask.new_tensor(1.0))
    torch.testing.assert_close(soft_derivative, hard_derivative)
    assert torch.isfinite(hard_derivative).all()
    assert torch.all(hard_derivative > 0.0)
    assert hard_derivative.max() < 1.0 / epsilon
    assert soft_selector.epsilon == epsilon
    assert soft_selector.hard_forward is False
    assert hard_selector.hard_forward is True


@pytest.mark.parametrize("invalid_width", [0.0, -0.1, math.inf, math.nan])
def test_invalid_epsilon_is_rejected(invalid_width):
    with pytest.raises(ValueError, match="epsilon must be finite and positive"):
        SmoothSelection(invalid_width)


@pytest.mark.parametrize("invalid_mode", [None, 0, 1, 0.03, "true"])
def test_non_boolean_hard_forward_is_rejected(invalid_mode):
    with pytest.raises(TypeError, match="hard_forward must be a bool"):
        SmoothSelection(0.1, invalid_mode)


@pytest.mark.parametrize("hard_forward", [False, True])
def test_zero_margins_return_the_all_selected_mask(hard_forward):
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)

    mask = SmoothSelection(0.1, hard_forward)(
        x, pos_margin=0.0, neg_margin=0.0
    )

    assert torch.equal(mask, torch.ones_like(x))


@pytest.mark.parametrize("hard_forward", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_mask_preserves_dtype_and_device(dtype, hard_forward):
    x = torch.tensor([-2.0, 0.0, 2.0], dtype=dtype)

    mask = SmoothSelection(0.1, hard_forward)(
        x, pos_margin=1.0, neg_margin=-1.0
    )

    assert mask.dtype == dtype
    assert mask.device == x.device
