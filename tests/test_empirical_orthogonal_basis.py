import math

import pytest
import torch

from src.orthogonal import (
    EmpiricalOrthogonalBasis,
    solve_transported_history,
)


def _generator(seed=0, device="cpu"):
    return torch.Generator(device=device).manual_seed(seed)


def test_empirical_orthonormality_and_exact_reconstruction():
    K = 64
    generator = _generator(1)
    columns = [
        torch.randn(K, generator=generator, dtype=torch.float64)
        for _ in range(5)
    ]
    factor = EmpiricalOrthogonalBasis(K, eps_rank=1e-12)

    for column in columns:
        factor.project_and_update(column)

    trajectory = torch.stack(columns, dim=1)
    empirical_gram = factor.B.T @ factor.B / K
    torch.testing.assert_close(
        empirical_gram,
        torch.eye(factor.rank, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        factor.B @ factor.Theta,
        trajectory,
        rtol=1e-11,
        atol=1e-11,
    )
    assert factor.orthogonality_error < 1e-12


def test_cgs2_residual_is_orthogonal_to_past_basis():
    K = 48
    generator = _generator(2)
    factor = EmpiricalOrthogonalBasis(K, eps_rank=1e-12)
    factor.project_and_update(
        torch.randn(K, generator=generator, dtype=torch.float64)
    )
    factor.project_and_update(
        torch.randn(K, generator=generator, dtype=torch.float64)
    )
    past_basis = factor.B.clone()
    result = factor.project_and_update(
        torch.randn(K, generator=generator, dtype=torch.float64)
    )

    orthogonality = past_basis.T @ result.residual / K
    torch.testing.assert_close(
        orthogonality,
        torch.zeros_like(orthogonality),
        rtol=0.0,
        atol=2e-15,
    )
    assert result.beta.shape == (2,)
    assert result.theta.shape == (3,)
    assert result.rank_increased


def test_empty_basis_handles_zero_initialization_then_grows_by_rank():
    K = 16
    factor = EmpiricalOrthogonalBasis(K, eps_rank=1e-12)

    zero_result = factor.project_and_update(torch.zeros(K))

    assert zero_result.beta.shape == (0,)
    assert zero_result.theta.shape == (0,)
    assert zero_result.rho.item() == 0.0
    assert not zero_result.rank_increased
    assert not zero_result.truncated
    assert factor.B.shape == (K, 0)
    assert factor.Theta.shape == (0, 1)

    column = torch.arange(1, K + 1, dtype=torch.float64)
    nonzero_result = factor.project_and_update(column)

    assert nonzero_result.rank_increased
    assert nonzero_result.rank_before == 0
    assert nonzero_result.rank_after == 1
    assert factor.Theta.shape == (1, 2)
    expected = torch.stack((torch.zeros(K), column), dim=1)
    torch.testing.assert_close(factor.B @ factor.Theta, expected)


def test_duplicate_column_does_not_use_time_as_basis_index():
    K = 32
    column = torch.randn(K, generator=_generator(3), dtype=torch.float64)
    factor = EmpiricalOrthogonalBasis(K, eps_rank=1e-12)
    first = factor.project_and_update(column)
    second = factor.project_and_update(column.clone())

    assert first.rank_increased
    assert not second.rank_increased
    assert second.rank_before == second.rank_after == 1
    assert factor.B.shape == (K, 1)
    assert factor.Theta.shape == (1, 2)
    torch.testing.assert_close(
        factor.B @ factor.Theta,
        torch.stack((column, column), dim=1),
        rtol=1e-12,
        atol=1e-12,
    )

    other_history = torch.stack(
        (
            torch.linspace(-1.0, 1.0, 13, dtype=torch.float64),
            torch.linspace(1.0, -1.0, 13, dtype=torch.float64),
        ),
        dim=1,
    )
    transported = solve_transported_history(other_history, factor.Theta)
    reference = other_history @ torch.linalg.pinv(factor.Theta)
    torch.testing.assert_close(transported, reference)


def _near_collinear_columns(K=80, relative_residual=1e-8):
    generator = _generator(4)
    first = torch.randn(K, generator=generator, dtype=torch.float64)
    direction = torch.randn(K, generator=generator, dtype=torch.float64)
    direction = direction - first * (torch.dot(first, direction) / torch.dot(first, first))
    direction = direction * (
        torch.linalg.vector_norm(first) / torch.linalg.vector_norm(direction)
    )
    second = first + relative_residual * direction
    return first, second


def test_near_collinear_direction_is_retained_below_rank_tolerance():
    K = 80
    first, second = _near_collinear_columns(K)
    factor = EmpiricalOrthogonalBasis(K, eps_rank=1e-10)
    factor.project_and_update(first)
    result = factor.project_and_update(second)

    assert result.rank_increased
    assert not result.truncated
    assert factor.rank == 2
    assert result.rho / result.input_norm > factor.eps_rank
    torch.testing.assert_close(
        factor.B @ factor.Theta,
        torch.stack((first, second), dim=1),
        rtol=1e-8,
        atol=1e-11,
    )


def test_rank_truncation_is_reported_as_regularization():
    K = 80
    first, second = _near_collinear_columns(K)
    factor = EmpiricalOrthogonalBasis(K, eps_rank=1e-6)
    factor.project_and_update(first)
    result = factor.project_and_update(second)

    assert not result.rank_increased
    assert result.truncated
    assert factor.rank == 1
    assert result.rho / result.input_norm <= factor.eps_rank

    trajectory = torch.stack((first, second), dim=1)
    reconstruction_error = trajectory - factor.B @ factor.Theta
    assert torch.linalg.vector_norm(reconstruction_error[:, 1]) > 0.0
    torch.testing.assert_close(
        reconstruction_error[:, 1], result.residual, rtol=1e-8, atol=1e-14
    )


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_and_device_are_preserved(device, dtype):
    K = 24
    factor = EmpiricalOrthogonalBasis(
        K, eps_rank=1e-6, dtype=dtype, device=device
    )
    column = torch.randn(
        K, generator=_generator(5, device=device), dtype=dtype, device=device
    )
    result = factor.project_and_update(column)

    assert factor.B.dtype == dtype
    assert factor.Theta.dtype == dtype
    assert factor.B.device.type == device
    assert factor.Theta.device.type == device
    assert result.beta.dtype == dtype
    assert result.residual.device.type == device
    expected_norm = math.sqrt(K)
    torch.testing.assert_close(
        torch.linalg.vector_norm(factor.B[:, 0]),
        torch.tensor(expected_norm, dtype=dtype, device=device),
        rtol=5e-6,
        atol=5e-6,
    )


def test_transported_history_matches_small_pseudoinverse_reference():
    K_source = 53
    K_other = 37
    n_columns = 4
    generator = _generator(6)
    factor = EmpiricalOrthogonalBasis(K_source, eps_rank=1e-12)
    for _ in range(n_columns):
        factor.project_and_update(
            torch.randn(K_source, generator=generator, dtype=torch.float64)
        )
    trajectory = torch.randn(
        (K_other, n_columns), generator=generator, dtype=torch.float64
    )

    transported = solve_transported_history(trajectory, factor.Theta)
    reference = trajectory @ torch.linalg.pinv(factor.Theta)

    assert transported.shape == (K_other, factor.rank)
    torch.testing.assert_close(
        transported, reference, rtol=1e-11, atol=1e-11
    )

    beta = torch.randn(factor.rank, generator=generator, dtype=torch.float64)
    alpha = torch.linalg.pinv(factor.Theta) @ beta
    torch.testing.assert_close(
        trajectory @ alpha,
        transported @ beta,
        rtol=1e-11,
        atol=1e-11,
    )


def test_transported_history_handles_empty_rank():
    trajectory = torch.randn(11, 1, dtype=torch.float64)
    coordinates = torch.empty(0, 1, dtype=torch.float64)

    transported = solve_transported_history(trajectory, coordinates)

    assert transported.shape == (11, 0)
    assert transported.dtype == trajectory.dtype
    assert transported.device == trajectory.device


def test_small_coordinate_diagnostics_are_lazy_and_finite():
    K = 40
    generator = _generator(7)
    factor = EmpiricalOrthogonalBasis(K, eps_rank=1e-12)
    for _ in range(3):
        factor.project_and_update(
            torch.randn(K, generator=generator, dtype=torch.float64)
        )

    singular_values = factor.coordinate_singular_values
    condition_number = factor.coordinate_condition_number

    assert singular_values.shape == (3,)
    assert torch.all(singular_values > 0.0)
    assert torch.isfinite(condition_number)
    assert condition_number >= 1.0
