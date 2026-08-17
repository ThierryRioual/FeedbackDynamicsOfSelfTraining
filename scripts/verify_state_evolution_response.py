"""Graph-free numerical checks for empirical orthogonal coordinates.

This script deliberately keeps Gram matrices and pseudoinverses on the
reference side only.  The implementation side uses sequential empirical CGS2
and the small least-squares transport implemented in :mod:`src.orthogonal`.

Run from the repository root, for example::

    PYTHONPATH=. python3 scripts/verify_state_evolution_response.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict

import torch

from src.orthogonal import EmpiricalOrthogonalBasis, solve_transported_history


@dataclass(frozen=True)
class WellConditionedReport:
    """Maximum absolute discrepancies in the defining coordinate identities."""

    errors: Dict[str, float]
    weight_orthogonality_error: float
    residual_orthogonality_error: float


@dataclass(frozen=True)
class IllConditionedReport:
    """Projection errors for an intentionally ill-conditioned trajectory."""

    trajectory_condition_number: float
    gram_condition_number: float
    coordinate_relative_error: float
    stable_lstsq_relative_error: float
    normal_equation_relative_error: float
    normal_equations_failed: bool


def _factor_trajectory(
    trajectory: torch.Tensor,
    *,
    eps_rank: float,
) -> EmpiricalOrthogonalBasis:
    basis = EmpiricalOrthogonalBasis(
        particle_count=trajectory.shape[0],
        eps_rank=eps_rank,
        dtype=trajectory.dtype,
        device=trajectory.device,
    )
    for column in trajectory.unbind(dim=1):
        result = basis.project_and_update(column)
        if result.truncated:
            raise RuntimeError(
                "the well-conditioned reference unexpectedly triggered "
                "numerical-rank truncation"
            )
    return basis


def _direct_projection_coefficients(
    history: torch.Tensor,
    current: torch.Tensor,
    particle_count: int,
) -> torch.Tensor:
    """Direct Gram/pseudoinverse formula, used only as a reference."""

    gram = (history.T @ history) / particle_count
    covariance = (history.T @ current) / particle_count
    return torch.linalg.pinv(gram) @ covariance


def _direct_memory_coefficients(
    trajectory: torch.Tensor,
    cross_history: torch.Tensor,
    orthogonal_residual: torch.Tensor,
    *,
    trajectory_particle_count: int,
    cross_particle_count: int,
) -> torch.Tensor:
    """Direct original-coordinate memory formula for reference checks."""

    gram = (trajectory.T @ trajectory) / trajectory_particle_count
    covariance = (cross_history.T @ orthogonal_residual) / cross_particle_count
    return torch.linalg.pinv(gram) @ covariance


def _maximum_absolute_error(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    return torch.max(torch.abs(left - right)).item()


def _record_close(
    errors: Dict[str, float],
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    errors[name] = _maximum_absolute_error(actual, expected)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@torch.no_grad()
def run_well_conditioned_check(
    *,
    K_w: int,
    K_g: int,
    n_past: int,
    delta: float,
    eps_rank: float,
    seed: int,
    atol: float,
    rtol: float,
) -> WellConditionedReport:
    """Compare coordinate formulas for ``q`` and ``p`` with direct formulas."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = torch.float64

    W_past = torch.randn((K_w, n_past), generator=generator, dtype=dtype)
    w_current = torch.randn(K_w, generator=generator, dtype=dtype)
    G_past = torch.randn((K_g, n_past), generator=generator, dtype=dtype)
    g_current = torch.randn(K_g, generator=generator, dtype=dtype)

    Q_past = torch.randn((K_g, n_past), generator=generator, dtype=dtype)
    P_past = torch.randn((K_w, n_past), generator=generator, dtype=dtype)
    z_g = torch.randn(K_g, generator=generator, dtype=dtype)
    z_w = torch.randn(K_w, generator=generator, dtype=dtype)

    weight_basis = _factor_trajectory(W_past, eps_rank=eps_rank)
    residual_basis = _factor_trajectory(G_past, eps_rank=eps_rank)

    B_w_past = weight_basis.B.clone()
    Theta_w_past = weight_basis.Theta.clone()
    B_g_past = residual_basis.B.clone()
    Theta_g_past = residual_basis.Theta.clone()

    Q_w_past = solve_transported_history(Q_past, Theta_w_past)
    P_g_past = solve_transported_history(P_past, Theta_g_past)

    weight_step = weight_basis.project_and_update(w_current)
    if weight_step.truncated:
        raise RuntimeError("the current weight unexpectedly triggered truncation")

    psi_w = (P_g_past.T @ weight_step.residual) / K_w
    q_coordinate = (
        Q_w_past @ weight_step.beta
        + (B_g_past @ psi_w) / math.sqrt(delta)
        + weight_step.rho * z_g
    )

    Q_current = torch.cat((Q_past, q_coordinate.unsqueeze(1)), dim=1)
    Q_w_current = solve_transported_history(Q_current, weight_basis.Theta)

    residual_step = residual_basis.project_and_update(g_current)
    if residual_step.truncated:
        raise RuntimeError("the current residual unexpectedly triggered truncation")

    psi_g = (Q_w_current.T @ residual_step.residual) / K_g
    p_coordinate = (
        P_g_past @ residual_step.beta
        + math.sqrt(delta) * (weight_basis.B @ psi_g)
        + residual_step.rho * z_w
    )

    # Direct reference formulas.  These intentionally form empirical Gram
    # matrices and their pseudoinverses; production code must not do so.
    alpha_w = _direct_projection_coefficients(W_past, w_current, K_w)
    w_perp_direct = w_current - W_past @ alpha_w
    v_w_direct = torch.sqrt(torch.mean(w_perp_direct.square()))
    phi_w = _direct_memory_coefficients(
        G_past,
        P_past,
        w_perp_direct,
        trajectory_particle_count=K_g,
        cross_particle_count=K_w,
    )
    q_direct = (
        Q_past @ alpha_w
        + (G_past @ phi_w) / math.sqrt(delta)
        + v_w_direct * z_g
    )

    alpha_g = _direct_projection_coefficients(G_past, g_current, K_g)
    g_perp_direct = g_current - G_past @ alpha_g
    v_g_direct = torch.sqrt(torch.mean(g_perp_direct.square()))
    W_current = torch.cat((W_past, w_current.unsqueeze(1)), dim=1)
    phi_g = _direct_memory_coefficients(
        W_current,
        Q_current,
        g_perp_direct,
        trajectory_particle_count=K_w,
        cross_particle_count=K_g,
    )
    p_direct = (
        P_past @ alpha_g
        + math.sqrt(delta) * (W_current @ phi_g)
        + v_g_direct * z_w
    )

    errors: Dict[str, float] = {}
    _record_close(
        errors,
        "weight reconstruction",
        weight_basis.B @ weight_basis.Theta,
        W_current,
        atol=atol,
        rtol=rtol,
    )
    G_current = torch.cat((G_past, g_current.unsqueeze(1)), dim=1)
    _record_close(
        errors,
        "residual reconstruction",
        residual_basis.B @ residual_basis.Theta,
        G_current,
        atol=atol,
        rtol=rtol,
    )
    _record_close(
        errors,
        "weight projection residual",
        weight_step.residual,
        w_perp_direct,
        atol=atol,
        rtol=rtol,
    )
    _record_close(
        errors,
        "residual projection residual",
        residual_step.residual,
        g_perp_direct,
        atol=atol,
        rtol=rtol,
    )
    _record_close(
        errors,
        "Q alpha = Q^[w] beta",
        Q_past @ alpha_w,
        Q_w_past @ weight_step.beta,
        atol=atol,
        rtol=rtol,
    )
    _record_close(
        errors,
        "P alpha = P^[g] beta",
        P_past @ alpha_g,
        P_g_past @ residual_step.beta,
        atol=atol,
        rtol=rtol,
    )
    _record_close(
        errors,
        "G phi = B^[g] psi",
        G_past @ phi_w,
        B_g_past @ psi_w,
        atol=atol,
        rtol=rtol,
    )
    _record_close(
        errors,
        "W phi = B^[w] psi",
        W_current @ phi_g,
        weight_basis.B @ psi_g,
        atol=atol,
        rtol=rtol,
    )
    _record_close(
        errors,
        "q fluctuation",
        q_coordinate,
        q_direct,
        atol=atol,
        rtol=rtol,
    )
    _record_close(
        errors,
        "p fluctuation",
        p_coordinate,
        p_direct,
        atol=atol,
        rtol=rtol,
    )

    return WellConditionedReport(
        errors=errors,
        weight_orthogonality_error=weight_basis.orthogonality_error.item(),
        residual_orthogonality_error=residual_basis.orthogonality_error.item(),
    )


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected)
    return (torch.linalg.vector_norm(actual - expected) / denominator).item()


@torch.no_grad()
def run_ill_conditioned_check(
    *,
    particle_count: int,
    rank: int,
    perturbation: float,
    seed: int,
) -> IllConditionedReport:
    """Compare CGS2 coordinates with a solve through squared normal equations."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = torch.float64

    raw = torch.randn((particle_count, rank), generator=generator, dtype=dtype)
    euclidean_basis = torch.linalg.qr(raw, mode="reduced").Q
    empirical_basis = math.sqrt(particle_count) * euclidean_basis

    # All columns share a dominant direction and differ only at scale
    # ``perturbation``.  Thus cond(A) is large and cond(A.T @ A) is squared.
    columns = [empirical_basis[:, 0]]
    columns.extend(
        empirical_basis[:, 0] + perturbation * empirical_basis[:, j]
        for j in range(1, rank)
    )
    trajectory = torch.stack(columns, dim=1)

    target_coefficients = torch.linspace(
        0.5, 1.5, rank, dtype=dtype
    )
    target = empirical_basis @ target_coefficients

    coordinate_basis = _factor_trajectory(trajectory, eps_rank=0.0)
    beta = (coordinate_basis.B.T @ target) / particle_count
    coordinate_projection = coordinate_basis.B @ beta

    stable_coefficients = torch.linalg.lstsq(
        trajectory,
        target,
        driver="gelsd",
    ).solution
    stable_projection = trajectory @ stable_coefficients

    empirical_gram = (trajectory.T @ trajectory) / particle_count
    empirical_covariance = (trajectory.T @ target) / particle_count
    normal_equations_failed = False
    try:
        normal_coefficients = torch.linalg.solve(
            empirical_gram,
            empirical_covariance,
        )
        normal_projection = trajectory @ normal_coefficients
        normal_error = _relative_error(normal_projection, target)
        if not math.isfinite(normal_error):
            normal_equations_failed = True
            normal_error = float("inf")
    except torch.linalg.LinAlgError:
        normal_equations_failed = True
        normal_error = float("inf")

    coordinate_error = _relative_error(coordinate_projection, target)
    stable_error = _relative_error(stable_projection, target)
    if not math.isfinite(coordinate_error):
        raise RuntimeError("the coordinate projection produced a non-finite error")
    if coordinate_error > 1e-5:
        raise RuntimeError(
            "CGS2 lost the ill-conditioned trajectory span: relative error "
            f"{coordinate_error:.3e}"
        )
    if not normal_equations_failed and normal_error <= coordinate_error:
        raise RuntimeError(
            "the chosen example did not expose normal-equation degradation: "
            f"coordinate={coordinate_error:.3e}, normal={normal_error:.3e}"
        )

    return IllConditionedReport(
        trajectory_condition_number=torch.linalg.cond(trajectory).item(),
        gram_condition_number=torch.linalg.cond(empirical_gram).item(),
        coordinate_relative_error=coordinate_error,
        stable_lstsq_relative_error=stable_error,
        normal_equation_relative_error=normal_error,
        normal_equations_failed=normal_equations_failed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-w", type=int, default=96)
    parser.add_argument("--k-g", type=int, default=137)
    parser.add_argument("--past-columns", type=int, default=4)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--eps-rank", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=2e-10)
    parser.add_argument("--rtol", type=float, default=2e-10)
    parser.add_argument("--ill-particles", type=int, default=128)
    parser.add_argument("--ill-rank", type=int, default=7)
    parser.add_argument("--ill-perturbation", type=float, default=1e-8)
    args = parser.parse_args()

    if args.k_w <= args.past_columns or args.k_g <= args.past_columns:
        parser.error("--k-w and --k-g must exceed --past-columns")
    if args.past_columns < 1:
        parser.error("--past-columns must be positive")
    if not math.isfinite(args.delta) or args.delta <= 0.0:
        parser.error("--delta must be finite and positive")
    if args.ill_rank < 2 or args.ill_particles < args.ill_rank:
        parser.error("require 2 <= --ill-rank <= --ill-particles")
    if not (0.0 < args.ill_perturbation < 1.0):
        parser.error("--ill-perturbation must lie in (0, 1)")
    return args


def main() -> None:
    args = parse_args()

    well = run_well_conditioned_check(
        K_w=args.k_w,
        K_g=args.k_g,
        n_past=args.past_columns,
        delta=args.delta,
        eps_rank=args.eps_rank,
        seed=args.seed,
        atol=args.atol,
        rtol=args.rtol,
    )
    print("Well-conditioned coordinate/direct agreement")
    for name, error in well.errors.items():
        print(f"  {name:<37} max abs error = {error:.3e}")
    print(
        "  weight basis orthogonality error      = "
        f"{well.weight_orthogonality_error:.3e}"
    )
    print(
        "  residual basis orthogonality error    = "
        f"{well.residual_orthogonality_error:.3e}"
    )

    ill = run_ill_conditioned_check(
        particle_count=args.ill_particles,
        rank=args.ill_rank,
        perturbation=args.ill_perturbation,
        seed=args.seed + 1,
    )
    print("\nIll-conditioned projection comparison")
    print(f"  cond(A)                               = {ill.trajectory_condition_number:.3e}")
    print(f"  cond(A.T @ A / K)                     = {ill.gram_condition_number:.3e}")
    print(f"  coordinate relative projection error = {ill.coordinate_relative_error:.3e}")
    print(f"  stable lstsq relative error           = {ill.stable_lstsq_relative_error:.3e}")
    if ill.normal_equations_failed:
        print("  normal-equation solve                 = failed/non-finite")
    else:
        print(
            "  normal-equation relative error         = "
            f"{ill.normal_equation_relative_error:.3e}"
        )


if __name__ == "__main__":
    main()
