"""Empirical-orthogonal coordinates for Monte Carlo trajectories.

The state-evolution particle inner product is ``x.T @ y / K``.  Consequently,
the columns of an empirical-orthonormal basis have ordinary Euclidean norm
``sqrt(K)`` rather than one.  This module keeps that normalization explicit and
contains no state-evolution model logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch


@dataclass(frozen=True)
class OrthogonalizationResult:
    """Result of projecting one trajectory column onto the past basis.

    ``beta`` contains coordinates in the basis that existed before the update.
    ``theta`` contains coordinates in the current, possibly augmented, basis.
    These vectors therefore differ by one entry whenever the numerical rank
    increases.
    """

    beta: torch.Tensor
    theta: torch.Tensor
    residual: torch.Tensor
    rho: torch.Tensor
    input_norm: torch.Tensor
    rank_before: int
    rank_after: int
    rank_increased: bool
    truncated: bool


@dataclass
class EmpiricalOrthogonalBasis:
    """Incrementally factor a particle trajectory using empirical CGS2.

    If no nonzero direction is truncated, the columns processed so far satisfy

    ``trajectory == basis @ coordinates``

    and ``basis.T @ basis / particle_count == I``.  A nonzero residual rejected
    by ``eps_rank`` is recorded as ``truncated=True``: it is numerical
    regularization, not an exact change of coordinates.
    """

    particle_count: int
    eps_rank: float = 1e-10
    dtype: torch.dtype = torch.float64
    device: Union[str, torch.device] = "cpu"

    residual_norms: List[torch.Tensor] = field(init=False, default_factory=list)
    input_norms: List[torch.Tensor] = field(init=False, default_factory=list)
    rank_history: List[int] = field(init=False, default_factory=list)
    rank_increased_history: List[bool] = field(
        init=False, default_factory=list
    )
    truncation_history: List[bool] = field(init=False, default_factory=list)

    _basis: torch.Tensor = field(init=False, repr=False)
    _coordinates: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.particle_count, bool)
            or not isinstance(self.particle_count, int)
            or self.particle_count <= 0
        ):
            raise ValueError(
                "particle_count must be a positive integer, "
                f"got {self.particle_count!r}"
            )
        self.eps_rank = float(self.eps_rank)
        if not math.isfinite(self.eps_rank) or self.eps_rank < 0.0:
            raise ValueError(
                "eps_rank must be finite and nonnegative, "
                f"got {self.eps_rank}"
            )
        if not torch.empty((), dtype=self.dtype).is_floating_point():
            raise TypeError(f"dtype must be floating point, got {self.dtype}")

        self.device = torch.device(self.device)
        self._basis = torch.empty(
            (self.particle_count, 0), dtype=self.dtype, device=self.device
        )
        self._coordinates = torch.empty(
            (0, 0), dtype=self.dtype, device=self.device
        )

    @property
    def basis(self) -> torch.Tensor:
        """Current basis ``B``, with ``B.T @ B / K == I``."""

        return self._basis

    @property
    def coordinates(self) -> torch.Tensor:
        """Current zero-padded coordinate matrix ``Theta``."""

        return self._coordinates

    # Short mathematical aliases are convenient in the state-evolution code.
    @property
    def B(self) -> torch.Tensor:
        return self._basis

    @property
    def Theta(self) -> torch.Tensor:
        return self._coordinates

    @property
    def numerical_rank(self) -> int:
        return self._basis.shape[1]

    @property
    def rank(self) -> int:
        return self.numerical_rank

    @property
    def n_columns(self) -> int:
        return self._coordinates.shape[1]

    def empirical_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``sqrt(x.T @ x / K)`` in this particle space."""

        x = self._validate_vector(x)
        return torch.sqrt(torch.mean(x.square()))

    def _validate_vector(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device).detach()
        if x.ndim != 1 or x.shape[0] != self.particle_count:
            raise ValueError(
                "trajectory column must have shape "
                f"({self.particle_count},), got {tuple(x.shape)}"
            )
        if not torch.isfinite(x).all():
            raise ValueError("trajectory column must contain only finite values")
        return x

    @torch.no_grad()
    def project_and_update(self, x: torch.Tensor) -> OrthogonalizationResult:
        """Project ``x`` on the past basis and update the factorization.

        Classical Gram--Schmidt is applied twice (CGS2).  The new direction is
        retained precisely when

        ``rho > eps_rank * empirical_norm(x)``.
        """

        x = self._validate_vector(x)
        rank_before = self.numerical_rank
        past_basis = self._basis

        if rank_before == 0:
            beta = x.new_empty((0,))
            residual = x.clone()
        else:
            c1 = (past_basis.T @ x) / self.particle_count
            u1 = x - past_basis @ c1
            c2 = (past_basis.T @ u1) / self.particle_count
            residual = u1 - past_basis @ c2
            beta = c1 + c2

        input_norm = torch.sqrt(torch.mean(x.square()))
        rho = torch.sqrt(torch.mean(residual.square()))
        rank_increased = bool(
            (rho > self.eps_rank * input_norm).item()
        )
        truncated = (not rank_increased) and bool((rho > 0.0).item())

        previous_coordinates = self._coordinates
        if rank_increased:
            new_direction = residual / rho
            self._basis = torch.cat(
                (past_basis, new_direction.unsqueeze(1)), dim=1
            )
            theta = torch.cat((beta, rho.reshape(1)))
            previous_coordinates = torch.cat(
                (
                    previous_coordinates,
                    previous_coordinates.new_zeros(
                        (1, previous_coordinates.shape[1])
                    ),
                ),
                dim=0,
            )
        else:
            theta = beta.clone()

        self._coordinates = torch.cat(
            (previous_coordinates, theta.unsqueeze(1)), dim=1
        )

        rank_after = self.numerical_rank
        self.residual_norms.append(rho.clone())
        self.input_norms.append(input_norm.clone())
        self.rank_history.append(rank_after)
        self.rank_increased_history.append(rank_increased)
        self.truncation_history.append(truncated)

        return OrthogonalizationResult(
            beta=beta.clone(),
            theta=theta.clone(),
            residual=residual.clone(),
            rho=rho.clone(),
            input_norm=input_norm.clone(),
            rank_before=rank_before,
            rank_after=rank_after,
            rank_increased=rank_increased,
            truncated=truncated,
        )

    @property
    @torch.no_grad()
    def orthogonality_error(self) -> torch.Tensor:
        """Spectral norm of ``B.T @ B / K - I`` (zero for an empty basis)."""

        if self.numerical_rank == 0:
            return torch.zeros((), dtype=self.dtype, device=self.device)
        gram = (self._basis.T @ self._basis) / self.particle_count
        identity = torch.eye(
            self.numerical_rank, dtype=self.dtype, device=self.device
        )
        return torch.linalg.matrix_norm(gram - identity, ord=2)

    @property
    @torch.no_grad()
    def coordinate_singular_values(self) -> torch.Tensor:
        """Singular values of the small coordinate matrix, on demand."""

        if self.numerical_rank == 0:
            return torch.empty((0,), dtype=self.dtype, device=self.device)
        return torch.linalg.svdvals(self._coordinates)

    @property
    @torch.no_grad()
    def coordinate_condition_number(self) -> torch.Tensor:
        """Condition number of ``Theta`` on its maintained row space."""

        singular_values = self.coordinate_singular_values
        if singular_values.numel() == 0:
            return torch.full(
                (), float("nan"), dtype=self.dtype, device=self.device
            )
        smallest = singular_values[-1]
        return torch.where(
            smallest > 0.0,
            singular_values[0] / smallest,
            torch.full_like(smallest, float("inf")),
        )


@torch.no_grad()
def solve_transported_history(
    trajectory: torch.Tensor,
    coordinates: torch.Tensor,
    rcond: Optional[float] = None,
    driver: Optional[str] = None,
) -> torch.Tensor:
    """Compute ``trajectory @ pinv(coordinates)`` by a small least-squares solve.

    If ``trajectory`` has shape ``(K_other, s)`` and ``coordinates`` has shape
    ``(r, s)``, this solves

    ``coordinates.T @ X ~= trajectory.T``

    and returns ``X.T`` with shape ``(K_other, r)``.  No Gram matrix or explicit
    pseudoinverse is formed.  The empty-rank result has shape ``(K_other, 0)``.
    """

    if trajectory.ndim != 2 or coordinates.ndim != 2:
        raise ValueError("trajectory and coordinates must both be matrices")
    if trajectory.shape[1] != coordinates.shape[1]:
        raise ValueError(
            "trajectory and coordinates must have the same number of columns, "
            f"got {trajectory.shape[1]} and {coordinates.shape[1]}"
        )
    if trajectory.dtype != coordinates.dtype:
        raise TypeError(
            "trajectory and coordinates must have the same dtype, "
            f"got {trajectory.dtype} and {coordinates.dtype}"
        )
    if trajectory.device != coordinates.device:
        raise ValueError(
            "trajectory and coordinates must be on the same device, "
            f"got {trajectory.device} and {coordinates.device}"
        )
    if not trajectory.is_floating_point() or not coordinates.is_floating_point():
        raise TypeError("trajectory and coordinates must be floating point")
    if not torch.isfinite(trajectory).all() or not torch.isfinite(coordinates).all():
        raise ValueError("trajectory and coordinates must contain only finite values")

    numerical_rank = coordinates.shape[0]
    if numerical_rank == 0:
        return trajectory.new_empty((trajectory.shape[0], 0))
    if coordinates.shape[1] == 0:
        raise ValueError("nonempty coordinate rank requires at least one column")
    if numerical_rank > coordinates.shape[1]:
        raise ValueError(
            "coordinate rank cannot exceed the number of trajectory columns, "
            f"got shape {tuple(coordinates.shape)}"
        )

    kwargs = {"rcond": rcond}
    if driver is not None:
        kwargs["driver"] = driver
    solution = torch.linalg.lstsq(
        coordinates.T,
        trajectory.T,
        **kwargs,
    ).solution
    return solution.T.contiguous()
