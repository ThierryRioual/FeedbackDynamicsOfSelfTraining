import pytest
import torch

from src.objectives import LogisticLoss
from src.utils import (
    compute_abstract_pseudo_residual_from,
    compute_population_error_from,
)


def _pseudo_residual(
    preactivation,
    *,
    label=None,
    indicator=None,
    selection_mask=None,
    selection_rate=1.0,
    coef=1.0,
    rho=0.5,
    time_index=None,
    initial_pseudo_label=None,
):
    if label is None:
        label = torch.ones_like(preactivation)
    if indicator is None:
        indicator = torch.zeros_like(preactivation)
    if selection_mask is None:
        selection_mask = torch.ones_like(preactivation)
    return compute_abstract_pseudo_residual_from(
        preactivation=preactivation,
        label=label,
        indicator=indicator,
        selection_mask=selection_mask,
        selection_rate=selection_rate,
        coef=coef,
        rho=rho,
        eta=0.1,
        loss_function=LogisticLoss(),
        time_index=time_index,
        initial_pseudo_label=initial_pseudo_label,
    )


def test_omitted_time_index_preserves_sign_pseudo_labels():
    preactivation = torch.tensor([2.0, -2.0], dtype=torch.float64)

    actual = _pseudo_residual(preactivation)

    pseudo_label = torch.tensor([1.0, -1.0], dtype=torch.float64)
    expected = (
        -0.1
        * (1.0 / (1.0 - 0.5))
        * LogisticLoss().gradient(preactivation, pseudo_label)
    )
    torch.testing.assert_close(actual, expected)


def test_active_first_update_requires_initial_pseudo_labels():
    with pytest.raises(ValueError, match="initial_pseudo_label is required"):
        _pseudo_residual(
            torch.tensor([1.0, -1.0], dtype=torch.float64),
            time_index=0,
        )


def test_first_update_uses_explicit_initial_pseudo_labels():
    preactivation = torch.tensor([2.0, -2.0], dtype=torch.float64)
    initial_pseudo_label = torch.tensor([-1.0, 1.0], dtype=torch.float64)

    actual = _pseudo_residual(
        preactivation,
        time_index=0,
        initial_pseudo_label=initial_pseudo_label,
    )

    expected = (
        -0.1
        * (1.0 / (1.0 - 0.5))
        * LogisticLoss().gradient(preactivation, initial_pseudo_label)
    )
    legacy = _pseudo_residual(preactivation)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, legacy)


def test_later_update_uses_sign_even_if_initial_labels_are_supplied():
    preactivation = torch.tensor([2.0, -2.0], dtype=torch.float64)
    initial_pseudo_label = torch.tensor([-1.0, 1.0], dtype=torch.float64)

    actual = _pseudo_residual(
        preactivation,
        time_index=1,
        initial_pseudo_label=initial_pseudo_label,
    )

    torch.testing.assert_close(actual, _pseudo_residual(preactivation))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coef": 0.0, "rho": 0.5},
        {"coef": 1.0, "rho": 1.0},
        {"coef": 1.0, "rho": 0.5, "selection_rate": 0.0},
    ],
)
def test_inactive_unlabeled_term_does_not_require_initial_labels(kwargs):
    preactivation = torch.tensor([0.4, -0.2], dtype=torch.float64)
    label = torch.tensor([1.0, -1.0], dtype=torch.float64)
    indicator = torch.ones_like(preactivation)

    actual = _pseudo_residual(
        preactivation,
        label=label,
        indicator=indicator,
        time_index=0,
        **kwargs,
    )

    expected = (
        -0.1
        * (indicator / kwargs["rho"])
        * LogisticLoss().gradient(preactivation, label)
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "initial_pseudo_label, exception, match",
    [
        (torch.tensor([1.0]), ValueError, "same shape"),
        (torch.tensor([1.0, 0.0]), ValueError, "belong to"),
        ([1.0, -1.0], TypeError, "torch.Tensor"),
    ],
)
def test_first_update_validates_initial_pseudo_labels(
    initial_pseudo_label, exception, match
):
    with pytest.raises(exception, match=match):
        _pseudo_residual(
            torch.tensor([1.0, -1.0]),
            time_index=0,
            initial_pseudo_label=initial_pseudo_label,
        )


@pytest.mark.parametrize(
    "time_index, exception",
    [(-1, ValueError), (0.0, TypeError), (True, TypeError)],
)
def test_time_index_must_be_a_nonnegative_integer_or_none(
    time_index, exception
):
    with pytest.raises(exception, match="time_index"):
        _pseudo_residual(
            torch.tensor([1.0, -1.0]),
            time_index=time_index,
            initial_pseudo_label=torch.tensor([1.0, -1.0]),
        )


def test_pseudo_residual_validates_core_shapes_and_true_labels():
    preactivation = torch.tensor([1.0, -1.0])
    with pytest.raises(ValueError, match="label and preactivation"):
        _pseudo_residual(preactivation, label=torch.ones(1))
    with pytest.raises(ValueError, match="selection_mask and preactivation"):
        _pseudo_residual(preactivation, selection_mask=torch.ones(1))
    with pytest.raises(ValueError, match="indicator"):
        _pseudo_residual(preactivation, indicator=torch.ones(3))
    with pytest.raises(ValueError, match="label entries"):
        _pseudo_residual(preactivation, label=torch.tensor([1.0, 0.0]))


@pytest.mark.parametrize(
    "b, m, p, expected",
    [
        (0.0, 0.0, 0.2, 0.8),
        (-1.0, 0.0, 0.2, 0.2),
        (1.0, 0.0, 0.2, 0.8),
        (0.0, 1.0, 0.2, 0.0),
        (0.0, -1.0, 0.2, 1.0),
    ],
)
def test_population_error_at_zero_effective_noise_respects_positive_tie_break(
    b, m, p, expected
):
    assert compute_population_error_from(b, m, 0.0, 1.0, p) == pytest.approx(
        expected
    )
    assert compute_population_error_from(b, m, 2.0, 0.0, p) == pytest.approx(
        expected
    )


def test_population_error_nonzero_noise_formula_is_unchanged():
    error = compute_population_error_from(
        b=0.3,
        m=0.8,
        tau=1.2,
        sigma=0.7,
        p=0.4,
    )

    assert error == pytest.approx(0.20357669433354122)
