"""Monte Carlo implementation of the effective self-training dynamics.

The production recursion in this module is entirely projection based.  It
maintains empirical-orthonormal bases for the weight and pseudo-residual
trajectories and evaluates the memory terms in those coordinates.  In
particular, it neither forms normal-equation inverses nor differentiates the
stochastic trajectory with autograd.
"""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.config import AlgorithmConfig, DataConfig
from src.orthogonal import EmpiricalOrthogonalBasis, solve_transported_history
from src.utils import (
    compute_abstract_pseudo_residual_from,
    compute_population_error_from,
)


DeviceLike = Union[str, torch.device]
ParameterBaseSampler = Callable[
    [int, torch.Generator, torch.dtype, torch.device],
    Tuple[torch.Tensor, torch.Tensor],
]
SampleBaseSampler = Callable[
    [int, torch.Generator, torch.dtype, torch.device],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


@dataclass(frozen=True)
class TheoremTrajectoryView:
    """The theorem-facing trajectory, excluding auxiliary terminal ``g^T,p^T``."""

    W: Tuple[torch.Tensor, ...]
    Q: Tuple[torch.Tensor, ...]
    R: Tuple[torch.Tensor, ...]
    G: Tuple[torch.Tensor, ...]
    P: Tuple[torch.Tensor, ...]


@dataclass
class MacroscopicStateEvolution:
    """Numerically integrate the effective dynamics by particle averaging.

    ``K`` is retained as a legacy alias for equal particle populations.  New
    code may set ``K_w`` and ``K_g`` independently; they are Monte Carlo
    budgets and have no prescribed relation to the limiting aspect ratio
    ``delta``.

    The optional base samplers implement the two joint limiting laws.  Their
    signatures are

    ``parameter_base_sampler(K_w, generator, dtype, device) -> (mu, w0)``

    and

    ``sample_base_sampler(K_g, generator, dtype, device) -> (Y, Delta, Y_init)``.

    A sampler must use the supplied generator for randomness.  This preserves
    dependence within each returned block while keeping the two base blocks and
    the two Gaussian innovation families independent.  If the samplers are
    omitted, the historical product-law initialization is used.  That legacy
    sample law has no exogenous ``Y_init`` and is therefore accepted at the
    first update only when the pseudo-labeled coefficient is zero.

    Bias handling remains single-sourced by ``algo_cfg.include_bias``.
    """

    data_cfg: DataConfig
    algo_cfg: AlgorithmConfig

    mc_seed: int = 42
    K: Optional[int] = 1000
    initial_bias: Optional[float] = 0.0
    initial_weight: Optional[torch.Tensor] = None

    K_w: Optional[int] = None
    K_g: Optional[int] = None
    eps_rank: float = 1e-10
    lstsq_rcond: Optional[float] = None
    lstsq_driver: Optional[str] = None
    dtype: torch.dtype = torch.float64
    device: DeviceLike = "cpu"
    parameter_base_sampler: Optional[ParameterBaseSampler] = None
    sample_base_sampler: Optional[SampleBaseSampler] = None

    _current_t: int = field(init=False, default=0)
    _debug: bool = False
    _debug_id_map: Dict[int, str] = field(init=False, default_factory=dict)

    signal: torch.Tensor = field(init=False)
    label: torch.Tensor = field(init=False)
    indicator: torch.Tensor = field(init=False)
    initial_pseudo_label: Optional[torch.Tensor] = field(init=False)

    bias: List[Optional[torch.Tensor]] = field(init=False)
    weight: List[Optional[torch.Tensor]] = field(init=False)
    preactivation: List[Optional[torch.Tensor]] = field(init=False)
    residual: List[Optional[torch.Tensor]] = field(init=False)

    # Compatibility names: forward_noise is q and backward_noise is p.
    forward_noise: List[Optional[torch.Tensor]] = field(init=False)
    backward_noise: List[Optional[torch.Tensor]] = field(init=False)
    sample_innovations: List[torch.Tensor] = field(init=False)
    parameter_innovations: List[torch.Tensor] = field(init=False)

    # Raw-trajectory memory coefficients phi, retained for diagnostics/API
    # compatibility only.  Production updates use the psi coordinates below.
    weight_memory: List[Optional[torch.Tensor]] = field(init=False)
    residual_memory: List[Optional[torch.Tensor]] = field(init=False)

    weight_projection_coordinates: List[Optional[torch.Tensor]] = field(init=False)
    residual_projection_coordinates: List[Optional[torch.Tensor]] = field(init=False)
    weight_coordinates: List[Optional[torch.Tensor]] = field(init=False)
    residual_coordinates: List[Optional[torch.Tensor]] = field(init=False)
    weight_memory_coordinates: List[Optional[torch.Tensor]] = field(init=False)
    residual_memory_coordinates: List[Optional[torch.Tensor]] = field(init=False)

    forward_innovation_scale: List[Optional[torch.Tensor]] = field(init=False)
    backward_innovation_scale: List[Optional[torch.Tensor]] = field(init=False)
    weight_rank: List[Optional[int]] = field(init=False)
    residual_rank: List[Optional[int]] = field(init=False)
    weight_rank_truncated: List[Optional[bool]] = field(init=False)
    residual_rank_truncated: List[Optional[bool]] = field(init=False)
    weight_orthogonality_error: List[Optional[torch.Tensor]] = field(init=False)
    residual_orthogonality_error: List[Optional[torch.Tensor]] = field(init=False)
    weight_coordinate_singular_values: List[Optional[torch.Tensor]] = field(init=False)
    residual_coordinate_singular_values: List[Optional[torch.Tensor]] = field(init=False)
    weight_coordinate_condition_number: List[Optional[torch.Tensor]] = field(init=False)
    residual_coordinate_condition_number: List[Optional[torch.Tensor]] = field(init=False)

    weight_signal_alignments: List[Optional[torch.Tensor]] = field(init=False)
    label_residual_alignments: List[Optional[torch.Tensor]] = field(init=False)
    mean_residual: List[Optional[torch.Tensor]] = field(init=False)
    selection_rate: List[Optional[torch.Tensor]] = field(init=False)
    weight_norm: List[Optional[float]] = field(init=False)
    error: List[Optional[float]] = field(init=False)
    decay: List[Optional[torch.Tensor]] = field(init=False)

    weight_basis: EmpiricalOrthogonalBasis = field(init=False)
    residual_basis: EmpiricalOrthogonalBasis = field(init=False)
    transported_forward_history: torch.Tensor = field(init=False)
    transported_backward_history: torch.Tensor = field(init=False)

    _sample_innovation_generator: torch.Generator = field(init=False, repr=False)
    _parameter_innovation_generator: torch.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self._validate_numerical_configuration()
        self._resolve_particle_counts()

        parameter_generator = self._make_generator(11)
        sample_generator = self._make_generator(23)
        self._sample_innovation_generator = self._make_generator(37)
        self._parameter_innovation_generator = self._make_generator(53)

        self.signal, sampled_weight = self._sample_parameter_base(
            parameter_generator
        )
        (
            self.label,
            self.indicator,
            self.initial_pseudo_label,
        ) = self._sample_sample_base(sample_generator)

        initial_bias_value = 0.0 if self.initial_bias is None else self.initial_bias
        initial_bias = torch.as_tensor(
            initial_bias_value, dtype=self.dtype, device=self.device
        )
        if initial_bias.numel() != 1 or not torch.isfinite(initial_bias).all():
            raise ValueError("initial_bias must be a finite scalar")
        initial_bias = initial_bias.reshape(()).detach().clone()
        if not self.algo_cfg.include_bias:
            if initial_bias.item() != 0.0:
                raise ValueError(
                    "initial_bias must be zero when include_bias=False"
                )
            initial_bias.zero_()
        self.initial_bias = initial_bias
        self.initial_weight = sampled_weight

        length = self.T + 1
        self.bias = [None] * length
        self.weight = [None] * length
        self.bias[0] = initial_bias
        self.weight[0] = sampled_weight

        self.preactivation = [None] * length
        self.residual = [None] * length
        self.forward_noise = [None] * length
        self.backward_noise = [None] * length

        self.sample_innovations = [
            self._randn(self.K_g, self._sample_innovation_generator)
            for _ in range(length)
        ]
        self.parameter_innovations = [
            self._randn(self.K_w, self._parameter_innovation_generator)
            for _ in range(length)
        ]

        optional_tensor_histories = (
            "weight_memory",
            "residual_memory",
            "weight_projection_coordinates",
            "residual_projection_coordinates",
            "weight_coordinates",
            "residual_coordinates",
            "weight_memory_coordinates",
            "residual_memory_coordinates",
            "forward_innovation_scale",
            "backward_innovation_scale",
            "weight_orthogonality_error",
            "residual_orthogonality_error",
            "weight_coordinate_singular_values",
            "residual_coordinate_singular_values",
            "weight_coordinate_condition_number",
            "residual_coordinate_condition_number",
            "weight_signal_alignments",
            "label_residual_alignments",
            "mean_residual",
            "selection_rate",
            "decay",
        )
        for name in optional_tensor_histories:
            setattr(self, name, [None] * length)
        self.weight_rank = [None] * length
        self.residual_rank = [None] * length
        self.weight_rank_truncated = [None] * length
        self.residual_rank_truncated = [None] * length
        self.weight_norm = [None] * length
        self.error = [None] * length

        self.weight_basis = EmpiricalOrthogonalBasis(
            self.K_w,
            eps_rank=self.eps_rank,
            dtype=self.dtype,
            device=self.device,
        )
        self.residual_basis = EmpiricalOrthogonalBasis(
            self.K_g,
            eps_rank=self.eps_rank,
            dtype=self.dtype,
            device=self.device,
        )
        self.transported_forward_history = torch.empty(
            (self.K_g, 0), dtype=self.dtype, device=self.device
        )
        self.transported_backward_history = torch.empty(
            (self.K_w, 0), dtype=self.dtype, device=self.device
        )
        self._current_t = 0

    def _validate_numerical_configuration(self) -> None:
        if (
            isinstance(self.algo_cfg.n_iterations, bool)
            or not isinstance(self.algo_cfg.n_iterations, int)
            or self.algo_cfg.n_iterations <= 0
        ):
            raise ValueError("algo_cfg.n_iterations must be a positive integer")
        if isinstance(self.mc_seed, bool) or not isinstance(self.mc_seed, int):
            raise TypeError("mc_seed must be an integer")
        if self.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "state-evolution linear algebra requires float32 or float64, "
                f"got {self.dtype}"
            )
        self.eps_rank = float(self.eps_rank)
        if not math.isfinite(self.eps_rank) or self.eps_rank < 0.0:
            raise ValueError("eps_rank must be finite and nonnegative")
        if self.lstsq_rcond is not None:
            self.lstsq_rcond = float(self.lstsq_rcond)
            if not math.isfinite(self.lstsq_rcond) or self.lstsq_rcond < 0.0:
                raise ValueError("lstsq_rcond must be finite and nonnegative")
        if not math.isfinite(float(self.sigma)) or self.sigma <= 0.0:
            raise ValueError("data noise scale sigma must be finite and positive")
        if not math.isfinite(float(self.delta)) or self.delta <= 0.0:
            raise ValueError("aspect ratio delta must be finite and positive")
        if not math.isfinite(float(self.eta)):
            raise ValueError("step size eta must be finite")
        if not math.isfinite(float(self.algo_cfg.penalty_param)):
            raise ValueError("penalty parameter lambda must be finite")

    def _resolve_particle_counts(self) -> None:
        if self.K_w is None and self.K_g is None:
            if (
                isinstance(self.K, bool)
                or not isinstance(self.K, int)
                or self.K <= 0
            ):
                raise ValueError("K must be a positive integer")
            self.K_w = self.K
            self.K_g = self.K
        elif self.K_w is None:
            self.K_w = self.K_g
        elif self.K_g is None:
            self.K_g = self.K_w

        if (
            isinstance(self.K_w, bool)
            or not isinstance(self.K_w, int)
            or self.K_w <= 0
        ):
            raise ValueError("K_w must be a positive integer")
        if (
            isinstance(self.K_g, bool)
            or not isinstance(self.K_g, int)
            or self.K_g <= 0
        ):
            raise ValueError("K_g must be a positive integer")
        self.K = self.K_w if self.K_w == self.K_g else None

    def _seed(self, stream: int) -> int:
        modulus = 2**63 - 1
        return (int(self.mc_seed) + 1_000_003 * stream) % modulus

    def _make_generator(self, stream: int) -> torch.Generator:
        try:
            generator = torch.Generator(device=self.device)
        except RuntimeError as exc:
            raise ValueError(
                f"independent torch generators are unavailable on {self.device}"
            ) from exc
        generator.manual_seed(self._seed(stream))
        return generator

    def _randn(self, size: int, generator: torch.Generator) -> torch.Tensor:
        return torch.randn(
            size,
            generator=generator,
            dtype=self.dtype,
            device=self.device,
        )

    @staticmethod
    @contextmanager
    def _isolated_python_numpy_rng(seed: int):
        """Temporarily seed and then restore legacy Python/NumPy RNGs."""

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        random.seed(seed)
        np.random.seed(seed % (2**32))
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def _invoke_sampler(
        self,
        sampler: Callable,
        size: int,
        generator: torch.Generator,
        fallback_stream: int,
    ):
        # Isolate legacy/global torch RNG calls made inside a custom sampler.
        cuda_devices: List[int] = []
        if self.device.type == "cuda":
            cuda_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices), self._isolated_python_numpy_rng(
            self._seed(fallback_stream)
        ):
            fallback_seed = self._seed(fallback_stream)
            cpu_generator = torch.Generator(device="cpu").manual_seed(
                fallback_seed
            )
            torch.set_rng_state(cpu_generator.get_state())
            if self.device.type == "cuda":
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(fallback_seed)
            return sampler(size, generator, self.dtype, self.device)

    def _sample_legacy_signal(self) -> torch.Tensor:
        values: List[float] = []
        cuda_devices: List[int] = []
        if self.device.type == "cuda":
            cuda_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices), self._isolated_python_numpy_rng(
            self._seed(83)
        ):
            signal_seed = self._seed(83)
            cpu_generator = torch.Generator(device="cpu").manual_seed(
                signal_seed
            )
            torch.set_rng_state(cpu_generator.get_state())
            if self.device.type == "cuda":
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(signal_seed)
            for _ in range(self.K_w):
                value = torch.as_tensor(self.data_cfg.signal_law())
                if value.numel() != 1 or not torch.isfinite(value).all():
                    raise ValueError("signal_law must return a finite scalar")
                values.append(float(value.reshape(()).item()))
        return torch.tensor(values, dtype=self.dtype, device=self.device)

    def _vector(self, name: str, value: torch.Tensor, size: int) -> torch.Tensor:
        """
        Validate and convert a user-supplied vector to a detached tensor.
        """
        result = torch.as_tensor(value, dtype=self.dtype, device=self.device).detach()
        if result.ndim != 1 or result.shape[0] != size:
            raise ValueError(f"{name} must have shape ({size},), got {tuple(result.shape)}")
        if not torch.isfinite(result).all():
            raise ValueError(f"{name} must contain only finite values")
        return result.clone()

    def _sample_parameter_base(
        self, generator: torch.Generator
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample the joint law of (mu, w_init) for the effective dynamics.
        """
        if self.parameter_base_sampler is not None:
            if self.initial_weight is not None:
                raise ValueError(
                    "initial_weight cannot be combined with parameter_base_sampler; "
                    "the sampler defines the joint (mu, w_init) law"
                )
            sampled = self._invoke_sampler(
                self.parameter_base_sampler,
                self.K_w,
                generator,
                fallback_stream=71,
            )
            if not isinstance(sampled, (tuple, list)) or len(sampled) != 2:
                raise ValueError("parameter_base_sampler must return (mu, w_init)")
            signal = self._vector("mu", sampled[0], self.K_w)
            weight = self._vector("w_init", sampled[1], self.K_w)
            return signal, weight

        signal = self._sample_legacy_signal()
        if self.initial_weight is None:
            weight = self._randn(self.K_w, generator)
        else:
            weight = self._vector("initial_weight", self.initial_weight, self.K_w)
        return signal, weight

    def _sample_sample_base(
        self, generator: torch.Generator
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample the joint law of (Y, Delta, Y_init) for the effective dynamics.
        """
        if self.sample_base_sampler is not None:
            sampled = self._invoke_sampler(
                self.sample_base_sampler,
                self.K_g,
                generator,
                fallback_stream=97,
            )
            if not isinstance(sampled, (tuple, list)) or len(sampled) != 3:
                raise ValueError(
                    "sample_base_sampler must return (Y, Delta, Y_init)"
                )
            label = self._vector("Y", sampled[0], self.K_g)
            indicator = self._vector("Delta", sampled[1], self.K_g)
            initial_label = self._vector("Y_init", sampled[2], self.K_g)
            self._validate_discrete_sample_base(label, indicator, initial_label)
            return label, indicator, initial_label

        label = (
            (
                torch.rand(
                    self.K_g,
                    generator=generator,
                    dtype=self.dtype,
                    device=self.device,
                )
                < self.p
            ).to(self.dtype)
            * 2.0
            - 1.0
        )
        indicator = (
            torch.rand(
                self.K_g,
                generator=generator,
                dtype=self.dtype,
                device=self.device,
            )
            < self.rho
        ).to(self.dtype)
        if self._pseudo_weight(0) != 0.0 and self.rho < 1.0:
            raise ValueError(
                "an explicit sample_base_sampler returning exogenous Y_init is "
                "required when the t=0 pseudo-labeled coefficient is nonzero"
            )
        return label, indicator, None

    def _validate_discrete_sample_base(
        self,
        label: torch.Tensor,
        indicator: torch.Tensor,
        initial_label: torch.Tensor,
    ) -> None:
        if not torch.all((label == -1.0) | (label == 1.0)):
            raise ValueError("Y particles must lie in {-1, +1}")
        if not torch.all((indicator == 0.0) | (indicator == 1.0)):
            raise ValueError("Delta particles must lie in {0, 1}")
        if not torch.all((initial_label == -1.0) | (initial_label == 1.0)):
            raise ValueError("Y_init particles must lie in {-1, +1}")
        disagreement = (indicator == 1.0) & (initial_label != label)
        if torch.any(disagreement):
            raise ValueError("Y_init must equal Y on every labeled particle")

    @property
    def T(self) -> int:
        return self.algo_cfg.n_iterations

    @property
    def rho(self) -> float:
        return self.data_cfg.supervision_ratio

    @property
    def p(self) -> float:
        return self.data_cfg.label_prior

    @property
    def sigma(self) -> float:
        return self.data_cfg.scale

    @property
    def delta(self) -> float:
        return self.data_cfg.data_to_dimension_ratio

    @property
    def eta(self) -> float:
        return self.algo_cfg.step_size

    @property
    def Q_w(self) -> torch.Tensor:
        """Current transported forward history ``Q^[w]``."""

        return self.transported_forward_history

    @property
    def P_g(self) -> torch.Tensor:
        """Current transported backward history ``P^[g]``."""

        return self.transported_backward_history

    @property
    def z_g(self) -> List[torch.Tensor]:
        """Sample-space innovations; these drive the forward fluctuations q."""

        return self.sample_innovations

    @property
    def z_w(self) -> List[torch.Tensor]:
        """Parameter-space innovations; these drive the backward fluctuations p."""

        return self.parameter_innovations

    @property
    def B_w(self) -> torch.Tensor:
        return self.weight_basis.B

    @property
    def Theta_w(self) -> torch.Tensor:
        return self.weight_basis.Theta

    @property
    def B_g(self) -> torch.Tensor:
        return self.residual_basis.B

    @property
    def Theta_g(self) -> torch.Tensor:
        return self.residual_basis.Theta

    @property
    def W_ring(self) -> torch.Tensor:
        """Available raw weight trajectory ``[w^0, ..., w^t]``."""

        return self._stack(
            self.weight, self._history_length(self.weight), self.K_w
        )

    @property
    def G_ring(self) -> torch.Tensor:
        """Available raw pseudo-residual trajectory ``[g^0, ..., g^t]``."""

        return self._stack(
            self.residual, self._history_length(self.residual), self.K_g
        )

    @property
    def Q_ring(self) -> torch.Tensor:
        """Available raw forward-fluctuation trajectory ``[q^0, ..., q^t]``."""

        return self._stack(
            self.forward_noise,
            self._history_length(self.forward_noise),
            self.K_g,
        )

    @property
    def P_ring(self) -> torch.Tensor:
        """Available raw backward-fluctuation trajectory ``[p^0, ..., p^t]``."""

        return self._stack(
            self.backward_noise,
            self._history_length(self.backward_noise),
            self.K_w,
        )

    @property
    def theorem_trajectory(self) -> TheoremTrajectoryView:
        """Expose ``(W_T,Q_T,R_T,G_{T-1},P_{T-1})`` without deleting diagnostics."""

        if any(value is None for value in self.weight[: self.T + 1]) or any(value is None for value in self.forward_noise[: self.T + 1]) or any(value is None for value in self.preactivation[: self.T + 1]):
            raise RuntimeError("compute_trajectory must be called before requesting theorem_trajectory")
        if any(value is None for value in self.residual[: self.T]) or any(value is None for value in self.backward_noise[: self.T]):
            raise RuntimeError("theorem residual histories are incomplete")
        return TheoremTrajectoryView(
            W=tuple(self.weight[: self.T + 1]),
            Q=tuple(self.forward_noise[: self.T + 1]),
            R=tuple(self.preactivation[: self.T + 1]),
            G=tuple(self.residual[: self.T]),
            P=tuple(self.backward_noise[: self.T]),
        )

    def _pseudo_weight(self, t: int) -> float:
        # The repository's schedules are retained as a documented experimental
        # extension.  The terminal diagnostic uses the final update's value.
        if self.T == 0:
            return 0.0
        return self.algo_cfg.get_pseudo_label_weight(min(t, self.T - 1))

    def _check_time_access(self, t: int) -> None:
        if not isinstance(t, int) or t < 0 or t > self._current_t or t > self.T:
            raise ValueError(
                f"Cannot access t={t}; current computed update index is {self._current_t}"
            )

    def _stack(self, history: List[Optional[torch.Tensor]], stop: int, rows: int) -> torch.Tensor:
        if stop == 0:
            return torch.empty((rows, 0), dtype=self.dtype, device=self.device)
        values = history[:stop]
        if any(value is None for value in values):
            raise RuntimeError("trajectory history contains an uncomputed column")
        return torch.stack(values, dim=1)

    @staticmethod
    def _history_length(history: List[Optional[torch.Tensor]]) -> int:
        count = 0
        while count < len(history) and history[count] is not None:
            count += 1
        if any(value is not None for value in history[count:]):
            raise RuntimeError("trajectory history is not contiguous")
        return count

    def _solve_transported(
        self, trajectory: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        return solve_transported_history(
            trajectory,
            coordinates,
            rcond=self.lstsq_rcond,
            driver=self.lstsq_driver,
        )

    def _solve_original_memory(
        self, coordinates: torch.Tensor, psi: torch.Tensor
    ) -> torch.Tensor:
        if coordinates.shape[1] == 0:
            return psi.new_empty((0,))
        if coordinates.shape[0] == 0:
            return psi.new_zeros((coordinates.shape[1],))
        kwargs = {"rcond": self.lstsq_rcond}
        if self.lstsq_driver is not None:
            kwargs["driver"] = self.lstsq_driver
        return torch.linalg.lstsq(coordinates, psi, **kwargs).solution.detach()

    @staticmethod
    def _condition_number_from(
        singular_values: torch.Tensor,
    ) -> torch.Tensor:
        if singular_values.numel() == 0:
            return singular_values.new_full((), float("nan"))
        smallest = singular_values[-1]
        return torch.where(
            smallest > 0.0,
            singular_values[0] / smallest,
            torch.full_like(smallest, float("inf")),
        )

    @torch.no_grad()
    def compute_trajectory(self) -> None:
        """Compute ``T`` updates and one terminal effective-process evaluation.

        The terminal pass constructs ``q^T``, ``g^T``, and ``p^T`` (hence the
        complete trajectory matrices through column ``T``) but deliberately
        does not create ``w^{T+1}`` or ``b^{T+1}``.  Thus ``n_iterations``
        retains its finite-algorithm meaning.
        """

        for t in range(self._current_t, self.T):
            self.step(t)
        if self.forward_noise[self.T] is None:
            self.forward_pass(self.T)
        if self.error[self.T] is None:
            self.compute_error(self.T)
        if self.backward_noise[self.T] is None:
            self._backward_pass(self.T, update_parameters=False)

    @torch.no_grad()
    def step(self, t: int) -> None:
        if t != self._current_t:
            raise RuntimeError(
                f"State mismatch: expected t={self._current_t}, received t={t}"
            )
        if t >= self.T:
            raise IndexError("Maximum number of parameter updates reached")
        self.forward_pass(t)
        self.compute_error(t)
        self._backward_pass(t, update_parameters=True)
        self._current_t += 1

    def get_decay_from(self, weight: torch.Tensor) -> torch.Tensor:
        return (
            -self.eta
            * self.algo_cfg.penalty_param
            * self.algo_cfg.penalty_function.gradient(weight)
        )

    @torch.no_grad()
    def forward_pass(self, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute the weight projection and forward half-step at time ``t``."""

        self._check_time_access(t)
        if self.preactivation[t] is not None and self.residual[t] is not None:
            return self.preactivation[t], self.residual[t]
        if self.preactivation[t] is not None or self.residual[t] is not None:
            raise RuntimeError("forward histories are only partially populated")
        if self.weight_basis.n_columns != t:
            raise RuntimeError("weight basis is not synchronized with time")
        current_weight = self.weight[t]
        if current_weight is None:
            raise RuntimeError(f"w^{t} has not been computed")

        past_residual_basis = self.residual_basis.B
        projection = self.weight_basis.project_and_update(current_weight)
        beta_w = projection.beta
        weight_perp = projection.residual
        v_w = projection.rho

        P_previous_g = self.transported_backward_history
        if P_previous_g.shape != (self.K_w, past_residual_basis.shape[1]):
            raise RuntimeError("P^[g] has inconsistent dimensions")
        psi_w = (P_previous_g.T @ weight_perp) / self.K_w

        Q_previous_w = self.transported_forward_history
        if Q_previous_w.shape != (self.K_g, beta_w.numel()):
            raise RuntimeError("Q^[w] has inconsistent dimensions")
        q = (
            Q_previous_w @ beta_w
            + (past_residual_basis @ psi_w) / math.sqrt(self.delta)
            + v_w * self.sample_innovations[t]
        )
        if q.shape != (self.K_g,):
            raise RuntimeError("q must live in sample-particle space")
        self.forward_noise[t] = q

        Q_trajectory = self._stack(self.forward_noise, t + 1, self.K_g)
        self.transported_forward_history = self._solve_transported(
            Q_trajectory, self.weight_basis.Theta
        )

        m = self.compute_weight_signal_alignment(t)
        self.compute_weight_norm(t)
        bias = self.bias[t]
        if bias is None:
            raise RuntimeError(f"b^{t} has not been computed")
        effective_bias = bias if self.algo_cfg.include_bias else bias.new_zeros(())
        r = effective_bias + m * self.label + self.sigma * q
        self.preactivation[t] = r
        omega = self.compute_selection_rate(t)
        pseudo_weight = self._pseudo_weight(t)
        g = self.compute_pseudo_residual_from(
            r,
            self.label,
            self.indicator,
            pseudo_weight,
            omega,
            time_index=t,
            initial_pseudo_label=self.initial_pseudo_label,
        )
        if g.shape != (self.K_g,):
            raise RuntimeError("g must live in sample-particle space")
        self.residual[t] = g

        self.weight_projection_coordinates[t] = beta_w
        self.weight_coordinates[t] = projection.theta
        self.weight_memory_coordinates[t] = psi_w
        self.forward_innovation_scale[t] = v_w
        self.weight_rank[t] = projection.rank_after
        self.weight_rank_truncated[t] = projection.truncated
        self.weight_orthogonality_error[t] = self.weight_basis.orthogonality_error
        singular_values = self.weight_basis.coordinate_singular_values
        self.weight_coordinate_singular_values[t] = singular_values
        self.weight_coordinate_condition_number[t] = self._condition_number_from(
            singular_values
        )
        return r, g

    @torch.no_grad()
    def backward_pass(self, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute the backward half-step and parameter update for ``t<T``."""

        if t >= self.T:
            raise IndexError("the terminal backward pass has no parameter update")
        return self._backward_pass(t, update_parameters=True)

    @torch.no_grad()
    def _backward_pass(
        self, t: int, update_parameters: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._check_time_access(t)
        if self.residual[t] is None:
            raise RuntimeError("forward_pass must be called before backward_pass")
        if self.residual_basis.n_columns == t:
            projection = self.residual_basis.project_and_update(self.residual[t])
            beta_g = projection.beta
            residual_perp = projection.residual
            v_g = projection.rho

            Q_current_w = self.transported_forward_history
            if Q_current_w.shape != (self.K_g, self.weight_basis.rank):
                raise RuntimeError("updated Q^[w] has inconsistent dimensions")
            psi_g = (Q_current_w.T @ residual_perp) / self.K_g

            P_previous_g = self.transported_backward_history
            if P_previous_g.shape != (self.K_w, beta_g.numel()):
                raise RuntimeError("past P^[g] has inconsistent dimensions")
            p = (
                P_previous_g @ beta_g
                + math.sqrt(self.delta) * (self.weight_basis.B @ psi_g)
                + v_g * self.parameter_innovations[t]
            )
            if p.shape != (self.K_w,):
                raise RuntimeError("p must live in parameter-particle space")
            self.backward_noise[t] = p

            P_trajectory = self._stack(self.backward_noise, t + 1, self.K_w)
            self.transported_backward_history = self._solve_transported(
                P_trajectory, self.residual_basis.Theta
            )

            self.residual_projection_coordinates[t] = beta_g
            self.residual_coordinates[t] = projection.theta
            self.residual_memory_coordinates[t] = psi_g
            self.backward_innovation_scale[t] = v_g
            self.residual_rank[t] = projection.rank_after
            self.residual_rank_truncated[t] = projection.truncated
            self.residual_orthogonality_error[t] = (
                self.residual_basis.orthogonality_error
            )
            singular_values = self.residual_basis.coordinate_singular_values
            self.residual_coordinate_singular_values[t] = singular_values
            self.residual_coordinate_condition_number[t] = (
                self._condition_number_from(singular_values)
            )
        elif not (
            self.residual_basis.n_columns == t + 1
            and self.backward_noise[t] is not None
        ):
            raise RuntimeError("residual basis is not synchronized with time")
        p = self.backward_noise[t]

        chi = self.compute_label_residual_alignments(t)
        zeta = self.compute_mean_residual(t)
        current_weight = self.weight[t]
        current_bias = self.bias[t]
        if self.decay[t] is None:
            self.decay[t] = self.get_decay_from(current_weight)
        h = self.decay[t]
        if not update_parameters:
            return current_bias, current_weight

        if self.bias[t + 1] is not None or self.weight[t + 1] is not None:
            if self.bias[t + 1] is None or self.weight[t + 1] is None:
                raise RuntimeError("parameter histories are only partially updated")
            return self.bias[t + 1], self.weight[t + 1]

        next_bias = (
            current_bias + zeta
            if self.algo_cfg.include_bias
            else current_bias.new_zeros(())
        )
        next_weight = (
            current_weight
            + h
            + chi * self.signal
            + (self.sigma / math.sqrt(self.delta)) * p
        )
        self.bias[t + 1] = next_bias
        self.weight[t + 1] = next_weight
        return next_bias, next_weight

    def compute_pseudo_residual_from(
        self,
        preactivation: torch.Tensor,
        label: torch.Tensor,
        indicator: torch.Tensor,
        coef: float,
        selection_rate: float,
        time_index: int = 1,
        initial_pseudo_label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        selection_mask = self.algo_cfg.selection_function(
            preactivation,
            self.algo_cfg.positive_margin,
            self.algo_cfg.negative_margin,
        )
        return compute_abstract_pseudo_residual_from(
            preactivation=preactivation,
            label=label,
            indicator=indicator,
            selection_mask=selection_mask,
            selection_rate=selection_rate,
            coef=coef,
            rho=self.rho,
            eta=self.eta,
            loss_function=self.algo_cfg.loss_function,
            time_index=time_index,
            initial_pseudo_label=initial_pseudo_label,
        )

    def compute_weight_signal_alignment(self, t: int) -> torch.Tensor:
        self._check_time_access(t)
        if self.weight_signal_alignments[t] is None:
            self.weight_signal_alignments[t] = (
                torch.dot(self.signal, self.weight[t]) / self.K_w
            ).detach()
        return self.weight_signal_alignments[t]

    def compute_label_residual_alignments(self, t: int) -> torch.Tensor:
        self._check_time_access(t)
        if self.residual[t] is None:
            raise RuntimeError(f"g^{t} has not been computed")
        if self.label_residual_alignments[t] is None:
            self.label_residual_alignments[t] = (
                torch.dot(self.label, self.residual[t]) / self.K_g
            ).detach()
        return self.label_residual_alignments[t]

    def compute_mean_residual(self, t: int) -> torch.Tensor:
        self._check_time_access(t)
        if self.residual[t] is None:
            raise RuntimeError(f"g^{t} has not been computed")
        if self.mean_residual[t] is None:
            self.mean_residual[t] = torch.mean(self.residual[t]).detach()
        return self.mean_residual[t]

    def compute_weight_norm(self, t: int) -> float:
        self._check_time_access(t)
        if self.weight_norm[t] is None:
            self.weight_norm[t] = torch.sqrt(
                torch.mean(self.weight[t].square())
            ).item()
        return self.weight_norm[t]

    def compute_selection_rate(self, t: int) -> torch.Tensor:
        """Return ``mean((1-Delta) S(r)) / (1-rho)`` as specified."""

        self._check_time_access(t)
        preactivation = self.preactivation[t]
        if preactivation is None or self.rho >= 1.0:
            omega = self.indicator.new_zeros(())
        else:
            mask = self.algo_cfg.selection_function(
                preactivation,
                self.algo_cfg.positive_margin,
                self.algo_cfg.negative_margin,
            )
            omega = torch.mean((1.0 - self.indicator) * mask) / (1.0 - self.rho)
        if not torch.isfinite(omega):
            raise FloatingPointError("selection-rate particle estimate is nonfinite")
        omega = omega.detach()
        self.selection_rate[t] = omega
        return omega

    def compute_error(self, t: int) -> float:
        self._check_time_access(t)
        if self.weight_signal_alignments[t] is None:
            self.compute_weight_signal_alignment(t)
        if self.weight_norm[t] is None:
            self.compute_weight_norm(t)
        bias = self.bias[t].item() if self.algo_cfg.include_bias else 0.0
        result = compute_population_error_from(
            bias,
            float(self.weight_signal_alignments[t]),
            float(self.weight_norm[t]),
            self.sigma,
            self.p,
        )
        self.error[t] = result
        return result

    @property
    def weight_grammian(self) -> torch.Tensor:
        """Small diagnostic Gram matrix; never used by the recursion."""

        stop = self.weight_basis.n_columns
        W = self._stack(self.weight, stop, self.K_w)
        return (W.T @ W / self.K_w).detach()

    @property
    def residual_grammian(self) -> torch.Tensor:
        """Small diagnostic Gram matrix; never used by the recursion."""

        stop = self.residual_basis.n_columns
        G = self._stack(self.residual, stop, self.K_g)
        return (G.T @ G / self.K_g).detach()

    def compute_weight_memory(self, t: int) -> torch.Tensor:
        """Lazily return diagnostic raw-column ``phi_t^[w]`` (no autograd).

        This is the exact original-trajectory coefficient when no nonzero
        direction has been truncated, and the coefficient of the maintained
        regularized coordinate model otherwise.  Production q uses ``psi``
        directly and does not call this method.
        """

        self._check_time_access(t)
        if self.weight_memory[t] is None:
            psi = self.weight_memory_coordinates[t]
            if psi is None:
                raise RuntimeError("forward_pass must be called first")
            past_theta = self.residual_basis.Theta[: psi.numel(), :t]
            self.weight_memory[t] = self._solve_original_memory(
                past_theta, psi
            )
        return self.weight_memory[t]

    def compute_residual_memory(self, t: int) -> torch.Tensor:
        """Lazily return diagnostic raw-column ``phi_t^[g]`` (no autograd).

        This is the exact original-trajectory coefficient when no nonzero
        direction has been truncated, and the coefficient of the maintained
        regularized coordinate model otherwise.  Production p uses ``psi``
        directly and does not call this method.
        """

        self._check_time_access(t)
        if self.residual_memory[t] is None:
            psi = self.residual_memory_coordinates[t]
            if psi is None:
                raise RuntimeError("backward_pass must be called first")
            current_theta = self.weight_basis.Theta[: psi.numel(), : t + 1]
            self.residual_memory[t] = self._solve_original_memory(
                current_theta, psi
            )
        return self.residual_memory[t]

    def compute_forward_projection_coef(self, t: int) -> torch.Tensor:
        """Return raw-column ``alpha_t^[w]`` lazily.

        After nonzero rank truncation this represents the maintained
        regularized coordinate model, not an exact change of raw coordinates.
        """

        self._check_time_access(t)
        beta = self.weight_projection_coordinates[t]
        if beta is None:
            raise RuntimeError("forward_pass must be called first")
        past_theta = self.weight_basis.Theta[: beta.numel(), :t]
        return self._solve_original_memory(past_theta, beta)

    def compute_residual_projection_coef(self, t: int) -> torch.Tensor:
        """Return raw-column ``alpha_t^[g]`` lazily.

        After nonzero rank truncation this represents the maintained
        regularized coordinate model, not an exact change of raw coordinates.
        """

        self._check_time_access(t)
        beta = self.residual_projection_coordinates[t]
        if beta is None:
            raise RuntimeError("backward_pass must be called first")
        past_theta = self.residual_basis.Theta[: beta.numel(), :t]
        return self._solve_original_memory(past_theta, beta)

    def compute_forward_noise_variance(self, t: int) -> torch.Tensor:
        self._check_time_access(t)
        scale = self.forward_innovation_scale[t]
        if scale is None:
            raise RuntimeError("forward_pass must be called first")
        return scale.square()

    def compute_backward_noise_variance(self, t: int) -> torch.Tensor:
        self._check_time_access(t)
        scale = self.backward_innovation_scale[t]
        if scale is None:
            raise RuntimeError("backward_pass must be called first")
        return scale.square()

    def compute_forward_noise(self, t: int) -> torch.Tensor:
        """Compatibility accessor for the already-computed ``q^t``."""

        self._check_time_access(t)
        if self.forward_noise[t] is None:
            self.forward_pass(t)
        return self.forward_noise[t]

    def compute_backward_noise(self, t: int) -> torch.Tensor:
        """Compatibility accessor for the already-computed ``p^t``."""

        self._check_time_access(t)
        if self.backward_noise[t] is None:
            self._backward_pass(t, update_parameters=False)
        return self.backward_noise[t]
