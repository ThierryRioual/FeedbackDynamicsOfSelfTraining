import math

import pytest
import torch

from src.objectives import HardSelection, LipschitzSelection


KAPPA = math.log(4.0)


def test_default_mode_restores_original_surrogate_forward_values():
    epsilon = 0.02
    x = torch.tensor(
        [
            -KAPPA - 0.01,
            -KAPPA + 0.01,
            KAPPA - 0.01,
            KAPPA + 0.01,
        ],
        dtype=torch.float64,
    )

    mask = LipschitzSelection(epsilon)(
        x, pos_margin=KAPPA, neg_margin=-KAPPA
    )

    torch.testing.assert_close(
        mask,
        torch.tensor([0.75, 0.25, 0.25, 0.75], dtype=x.dtype),
    )


def test_hard_forward_changes_values_but_not_surrogate_derivative():
    epsilon = 0.02
    values = torch.tensor(
        [
            -KAPPA - 0.01,
            -KAPPA + 0.01,
            KAPPA - 0.01,
            KAPPA + 0.01,
        ],
        dtype=torch.float64,
    )
    soft_input = values.clone().requires_grad_()
    hard_input = values.clone().requires_grad_()
    soft_selector = LipschitzSelection(epsilon, False)
    hard_selector = LipschitzSelection(epsilon, True)

    soft_mask = soft_selector(
        soft_input, pos_margin=KAPPA, neg_margin=-KAPPA
    )
    hard_mask = hard_selector(
        hard_input, pos_margin=KAPPA, neg_margin=-KAPPA
    )
    soft_derivative = torch.autograd.grad(soft_mask.sum(), soft_input)[0]
    hard_derivative = torch.autograd.grad(hard_mask.sum(), hard_input)[0]

    expected_hard = HardSelection()(
        hard_input, pos_margin=KAPPA, neg_margin=-KAPPA
    )
    assert torch.equal(hard_mask.detach(), expected_hard)
    assert torch.equal(
        hard_mask.detach(),
        torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=values.dtype),
    )
    expected_derivative = torch.tensor(
        [-25.0, -25.0, 25.0, 25.0], dtype=values.dtype
    )
    torch.testing.assert_close(soft_derivative, expected_derivative)
    torch.testing.assert_close(hard_derivative, expected_derivative)


def test_epsilon_controls_the_single_transition_width_and_slope():
    epsilon = 0.04
    x = torch.tensor([-KAPPA, KAPPA], dtype=torch.float64, requires_grad=True)
    selector = LipschitzSelection(epsilon)

    mask = selector(x, pos_margin=KAPPA, neg_margin=-KAPPA)
    derivative = torch.autograd.grad(mask.sum(), x)[0]

    torch.testing.assert_close(mask, torch.full_like(mask, 0.5))
    torch.testing.assert_close(
        derivative,
        torch.tensor(
            [-1.0 / (2.0 * epsilon), 1.0 / (2.0 * epsilon)],
            dtype=x.dtype,
        ),
    )
    assert selector.epsilon == epsilon


def test_surrogate_derivative_is_zero_outside_epsilon_bands():
    x = torch.tensor(
        [-KAPPA - 0.03, -KAPPA + 0.03, KAPPA - 0.03, KAPPA + 0.03],
        dtype=torch.float64,
        requires_grad=True,
    )
    mask = LipschitzSelection(0.02, True)(
        x, pos_margin=KAPPA, neg_margin=-KAPPA
    )

    derivative = torch.autograd.grad(mask.sum(), x)[0]

    torch.testing.assert_close(derivative, torch.zeros_like(derivative))


@pytest.mark.parametrize("invalid_width", [0.0, -0.1, math.inf, math.nan])
def test_invalid_epsilon_is_rejected(invalid_width):
    with pytest.raises(ValueError, match="epsilon must be finite and positive"):
        LipschitzSelection(invalid_width)


@pytest.mark.parametrize("invalid_mode", [None, 0, 1, 0.03, "true"])
def test_non_boolean_hard_forward_is_rejected(invalid_mode):
    with pytest.raises(TypeError, match="hard_forward must be a bool"):
        LipschitzSelection(0.1, invalid_mode)


@pytest.mark.parametrize("hard_forward", [False, True])
def test_zero_margins_return_the_all_selected_mask(hard_forward):
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)

    mask = LipschitzSelection(0.1, hard_forward)(
        x, pos_margin=0.0, neg_margin=0.0
    )

    assert torch.equal(mask, torch.ones_like(x))


@pytest.mark.parametrize("hard_forward", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_mask_preserves_dtype_and_device(dtype, hard_forward):
    x = torch.tensor([-2.0, 0.0, 2.0], dtype=dtype)

    mask = LipschitzSelection(0.1, hard_forward)(
        x, pos_margin=1.0, neg_margin=-1.0
    )

    assert mask.dtype == dtype
    assert mask.device == x.device
