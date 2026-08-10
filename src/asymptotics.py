from IPython.core import getipython
from IPython.core import getipython
from IPython.core import getipython
from IPython.core import getipython
import torch

from scipy.stats import norm

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional

from src.config import DataConfig, AlgorithmConfig
from src.utils import compute_projection_coef_from, compute_abstract_pseudo_residual_from, compute_population_error_from



@dataclass
class MacroscopicStateEvolution:
    """
    Numerically integrate the state evolution equations using Monte Carlo averaging.

    Bias handling is controlled exclusively by ``algo_cfg.include_bias``, as in
    ``SelfTrainedGradientDescent``. If it is false, the effective intercept is
    fixed at zero for the entire trajectory.
    """
    # Frozen configuration objects (same pattern as SelfTrainedGradientDescent / IsotropicGaussian)
    data_cfg: DataConfig
    algo_cfg: AlgorithmConfig

    mc_seed: int = field(default=42) # monte carlo seed
    K: int = field(default=1000) # number of monte carlo samples

    initial_bias: Optional[float] = field(default=0.0) # b_{init}
    initial_weight: torch.Tensor = field(default=None) # w_{init}

    _current_t: int = field(default=0) # current time step (for convenience)
    _debug: bool = field(default=False) # debugging flag
    _debug_id_map: Dict[int, str] = field(init=False) # mapping of tensor ids to names for debugging

    # Instance variables    
    signal: torch.Tensor = field(init=False) # \mu
    label: torch.Tensor = field(init=False) # Y
    indicator: torch.Tensor = field(init=False) # \Delta

    bias: List[torch.Tensor] = field(init=False) # b 
    weight: List[torch.Tensor] = field(init=False) # w

    preactivation: List[torch.Tensor] = field(init=False) # r (preactivation)
    residual: List[torch.Tensor] = field(init=False) # g (pseudo-residual)

    forward_noise: List[torch.Tensor] = field(init=False) # \omega
    backward_noise: List[torch.Tensor] = field(init=False) # \xi

    weight_memory: List[torch.Tensor] = field(init=False) # \gamma^{[w]}
    residual_memory: List[torch.Tensor] = field(init=False) # \gamma^{[g]}

    weight_signal_alignments: List[torch.Tensor] = field(init=False) # m
    label_residual_alignments: List[torch.Tensor] = field(init=False) # \chi
    mean_residual: List[torch.Tensor] = field(init=False) # \zeta
    selection_rate: List[torch.Tensor] = field(init=False) # A

    error: List[float] = field(init=False)

    def __post_init__(self):
        """
        Initialize the state evolution.
        """
        torch.manual_seed(self.mc_seed)

        if self._debug:
            self._debug_id_map = {}

        self.signal = torch.tensor([self.data_cfg.signal_law() for _ in range(self.K)]) # signal vector
        self.label = (torch.rand(self.K) < self.data_cfg.label_prior).double() * 2 - 1 # transform to {+1, -1}
        self.indicator = (torch.rand(self.K) < self.data_cfg.supervision_ratio).double() # observation indicator 

        initial_bias_value = (
            0.0 if self.initial_bias is None else self.initial_bias
        )
        initial_bias = torch.as_tensor(
            initial_bias_value,
            dtype=torch.get_default_dtype(),
        )
        if initial_bias.numel() != 1:
            raise ValueError("initial_bias must be scalar")
        initial_bias = initial_bias.reshape(())
        if not self.algo_cfg.include_bias:
            if initial_bias.item() != 0.0:
                raise ValueError(
                    "initial_bias must be zero when include_bias=False"
                )
            initial_bias = torch.zeros_like(initial_bias)
        self.initial_bias = (
            initial_bias.detach().clone().requires_grad_(
                self.algo_cfg.include_bias
            )
        )

        if self.initial_weight is None:
            self.initial_weight = torch.randn(self.K, requires_grad=True)
        else:
            self.initial_weight = torch.tensor(self.initial_weight, requires_grad=True)

        T = self.algo_cfg.n_iterations

        self.bias = [None] * (T + 1)
        self.weight = [None] * (T + 1)

        self.bias[0] = self.initial_bias
        self.weight[0] = self.initial_weight

        self.preactivation = [None] * (T + 1)
        self.residual = [None] * (T + 1)

        self.forward_noise = [None] * (T + 1)
        self.backward_noise = [None] * T

        self.weight_memory = [None] * (T + 1) # First element of weight_memory list is supposed to be None as that vector does not exist
        self.residual_memory = [None] * T

        self.weight_signal_alignments = [None] * (T + 1)
        self.label_residual_alignments = [None] * (T + 1)
        self.mean_residual = [None] * (T + 1)
        self.selection_rate = [None] * (T + 1)

        self.error = [None] * (T + 1)

        self._current_t = 0

    @property
    def T(self) -> int:
        """
        Returns the number of iterations.
        """
        return self.algo_cfg.n_iterations

    @property
    def rho(self) -> float:
        """
        Returns the observation prior.
        """
        return self.data_cfg.supervision_ratio

    @property
    def p(self) -> float:
        """
        Returns the label prior.
        """
        return self.data_cfg.label_prior

    @property
    def sigma(self) -> float:
        """
        Returns the signal strength.
        """
        return self.data_cfg.scale
    
    @property
    def delta(self) -> float:
        """
        Returns the data-to-dimension ratio.
        """
        return self.data_cfg.data_to_dimension_ratio

    @property
    def eta(self) -> float:
        """
        Returns the learning rate.
        """
        return self.algo_cfg.step_size

    def get_decay_from(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Returns the weight decay (penalty) gradient function: h = -η λ ∇P(w).
        """
        return - self.eta * self.algo_cfg.penalty_param * self.algo_cfg.penalty_function.gradient(weight)
    
    def compute_pseudo_residual_from(self, preactivation: torch.Tensor, 
            label: torch.Tensor, indicator: torch.Tensor,
            coef: float, selection_rate: float,
        ) -> torch.Tensor:
        """
        Computes the pseudo-residual g = -η ∇_r R(r).
        This is the gradient of the empirical risk with respect to the preactivation r, not the weights.
        """
        selection_mask = self.algo_cfg.selection_function(
            preactivation, 
            self.algo_cfg.positive_margin, 
            self.algo_cfg.negative_margin
        )

        return compute_abstract_pseudo_residual_from(
            preactivation=preactivation, label=label, indicator=indicator,
            selection_mask=selection_mask, selection_rate=selection_rate,
            coef=coef, rho=self.rho, 
            eta=self.eta, loss_function=self.algo_cfg.loss_function
        )
    
    def _check_time_access(self, t: int) -> None:
        """
        Ensures that helper methods do not access uncomputed future states.
        """
        if t > self._current_t:
            raise ValueError(f"Cannot access uncomputed future steps. Current t is {self._current_t}.")
        return None

    def compute_trajectory(self) -> None:
        """Computes the full macroscopic trajectory of the algorithm."""
        for t in range(self.T):
            self.step(t)

        # For the very last step to get macroscopic quantities at t=T
        self.forward_pass(self.T)
        self.compute_error(self.T)
        self.compute_label_residual_alignments(self.T)
        self.compute_mean_residual(self.T)
        return None

    def step(self, t: int) -> None:
        """
        Performs one full SE iteration at time t:
          1. Forward pass: compute field r^t and pseudo-residual g^t
          2. Backward pass: compute updated weight w^{t+1} and bias b^{t+1}
        """
        if t != self._current_t:
            raise RuntimeError(f"State mismatch: Expected step t={self._current_t}, but received t={t}. Did you call step multiple times or skip a step?")
        if t >= self.T:
            raise IndexError("Maximum iterations reached.")

        self.forward_pass(t) # \mathcal{F}_t -> \mathcal{F}_t^+
        self.compute_error(t)
        self.backward_pass(t) # \mathcal{F}_t^+ -> \mathcal{F}_{t+1}

        self._current_t += 1

        return None

    def forward_pass(self, t: int) -> Tuple[torch.Tensor, torch.Tensor]:   
        """
        Performs the forward pass of the self-training algorithm.
        Computes the field r and then the pseudo-residual g.
        """

        m = self.compute_weight_signal_alignment(t) # m
        omega = self.compute_forward_noise(t) # \omega
        b = (
            self.bias[t]
            if self.algo_cfg.include_bias
            else self.bias[t].new_zeros(())
        )
        ones = torch.ones_like(omega)

        pi = self.algo_cfg.get_pseudo_label_weight(min(t, self.T - 1)) # Handles the very last step 

        if t == 0: # Handles initial step t=0 (Empty sum)
            r = b * ones + m * self.label + omega

        else: # Handles t>0
            gamma = self.compute_weight_memory(t) # \gamma^{[w]}
            G = torch.stack(self.residual[:t], dim=1) # G_{t-1} (residual matrix)

            r = b * ones + m * self.label + (G @ gamma) / (self.delta ** 0.5) + omega # r^{t}

            if self._debug:
                self._debug_id_map[id(m)] = f"m_{t}"
                self._debug_id_map[id(ones)] = f"1_n"
                self._debug_id_map[id(gamma)] = f"gamma^w_{t}"

        self.preactivation[t] = r # r (preactivation)

        A = self.compute_selection_rate(t)
        g = self.compute_pseudo_residual_from(r, self.label, self.indicator, pi, A) # g^{t}
        self.residual[t] = g # g (pseudo-residual)

        return r, g

    def backward_pass(self, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the backward pass of the self-training algorithm.
        Computes the updated bias and weight using the pseudo-residual memory and the decay step.
        """

        gamma = self.compute_residual_memory(t) # \gamma^{[g]}
        chi = self.compute_label_residual_alignments(t) # \chi
        zeta = self.compute_mean_residual(t) # \zeta
        xi = self.compute_backward_noise(t) # \xi

        prev_b = self.bias[t]
        prev_w = self.weight[t]

        W = torch.stack(self.weight[:t+1], dim=1) # W_{t} (residual matrix)
        h = self.get_decay_from(prev_w) # h (weight decay)

        b = (
            prev_b + zeta
            if self.algo_cfg.include_bias
            else prev_b.new_zeros(())
        )
        self.bias[t+1] = b

        w = prev_w + h + chi * self.signal + W @ gamma + xi / (self.delta ** 0.5)
        self.weight[t+1] = w

        if self._debug:
            self._debug_id_map[id(h)] = f"h^{t+1}"
            self._debug_id_map[id(chi)] = f"chi_{t}"
            self._debug_id_map[id(self.delta)] = f"delta"
            self._debug_id_map[id(gamma)] = f"gamma^g_{t+1}"

        return b, w

    def compute_error(self, t: int) -> float:
        """
        Computes the error
        """
        self._check_time_access(t)
        b = self.bias[t].item() if self.algo_cfg.include_bias else 0.0
        m = self.weight_signal_alignments[t]
        tau = torch.sqrt(torch.mean(self.weight[t] ** 2)).item()
        err = compute_population_error_from(b, m, tau, self.sigma, self.p)
        self.error[t] = err
        return err

    @property
    def weight_grammian(self) -> torch.Tensor:
        """
        Computes the weight Gram matrix ($$ C^{[w]} $$)
        """
        W = torch.stack(self.weight[:self._current_t], dim=1)
        return ((W.T @ W) / self.K).detach()
    
    @property
    def residual_grammian(self) -> torch.Tensor:
        """
        Computes the pseudo-residual Gram matrix ($$ C^{[g]} $$)
        """
        G = torch.stack(self.residual[:self._current_t], dim=1)
        return ((G.T @ G) / self.K).detach()
    
    def compute_weight_memory(self, t: int) -> torch.Tensor:
        """
        Computes the weight memory vector ($$ \gamma^{[w]} $$)
        """
        self._check_time_access(t)
        current_weight = self.weight[t] # w^t
        noise_inputs = tuple(self.backward_noise[:t]) # Pass sequence of original tensors

        derivatives = torch.autograd.grad(
            outputs=current_weight,
            inputs=noise_inputs,
            grad_outputs=torch.ones_like(current_weight),
            retain_graph=True,
            allow_unused=True
        )

        mean_derivatives = []
        for d in derivatives:
            if d is None:
                mean_derivatives.append(0.0)
            else:
                mean_derivatives.append(torch.mean(d).item())
        
        gamma = torch.tensor(mean_derivatives, dtype=torch.float64).detach()
        self.weight_memory[t] = gamma

        return gamma

    def compute_residual_memory(self, t: int) -> torch.Tensor:
        """
        Computes the pseudo-residual memory vector ($$ \gamma^{[g]} $$).

        For each forward-noise input, the required quantity is the mean
        diagonal Jacobian, ``trace(J) / K``. A Rademacher trace probe computes
        ``z.T @ J @ z / K`` through one vector-Jacobian product. It is exact for
        the coordinatewise residual map used by state evolution and remains an
        unbiased trace estimate if a future change introduces particle coupling.
        """
        self._check_time_access(t)
        current_residual = self.residual[t] # g^t
        noise_inputs = tuple(self.forward_noise[:t+1]) # Pass sequence of original tensors

        trace_probe = self._response_trace_probe(current_residual, t)

        derivatives = torch.autograd.grad(
            outputs=current_residual,
            inputs=noise_inputs,
            grad_outputs=trace_probe,
            retain_graph=True,
            allow_unused=True
        )

        mean_derivatives = []
        for d in derivatives:
            if d is None:
                mean_derivatives.append(current_residual.new_zeros(()))
            else:
                if d.shape != trace_probe.shape:
                    raise RuntimeError(
                        "Residual and forward-noise particles must have matching "
                        "shapes to estimate the diagonal response."
                    )
                mean_derivatives.append(torch.mean(trace_probe * d))
        
        gamma = torch.stack(mean_derivatives).detach()
        self.residual_memory[t] = gamma

        return gamma

    def _response_trace_probe(
        self, reference: torch.Tensor, t: int
    ) -> torch.Tensor:
        """
        Returns a deterministic local Rademacher probe without changing RNG state.
        Implements the Hutchinson trace estimator for the diagonal Jacobian of the residual map.
        """
        generator = torch.Generator(device="cpu")
        seed = (int(self.mc_seed) + 104729 * (t + 1)) % (2**63 - 1)
        generator.manual_seed(seed)
        probe = torch.randint(
            low=0,
            high=2,
            size=reference.shape,
            generator=generator,
            dtype=torch.int8,
            device="cpu",
        )
        return (2.0 * probe - 1.0).to(
            dtype=reference.dtype, device=reference.device
        )

    def compute_weight_signal_alignment(self, t: int) -> float:
        """
        Computes the weight-signal alignment ($$ m $$)
        """
        self._check_time_access(t)
        w = self.weight[t] # w^t
        mu = self.signal
        m = (torch.dot(w, mu) / self.K).detach()
        self.weight_signal_alignments[t] = m
        return m

    def compute_label_residual_alignments(self, t: int) -> float:
        """
        Computes the label-residual covariance ($$ \chi $$)
        """
        self._check_time_access(t)
        g = self.residual[t] # g^t
        y = self.label
        chi = (torch.dot(g, y) / self.K).detach()
        self.label_residual_alignments[t] = chi
        return chi

    def compute_mean_residual(self, t: int) -> float:
        """
        Computes the mean pseudo-residual ($$ \zeta $$)
        """
        self._check_time_access(t)
        g = self.residual[t] # g^t
        ones_vec = torch.ones(self.K)
        zeta = (torch.dot(g, ones_vec) / self.K).detach()
        self.mean_residual[t] = zeta
        return zeta    

    def compute_forward_projection_coef(self, t: int) -> torch.Tensor:
        r"""
        Computes the forward projection coefficients ($$ \alpha^t $$).
        """
        self._check_time_access(t)
        W = torch.stack(self.weight[:t], dim=1)
        v = self.weight[t] - self.weight[t-1] if t > 0 else self.weight[t]
        return compute_projection_coef_from(W, v)

    def compute_residual_projection_coef(self, t: int) -> torch.Tensor:
        r"""
        Computes the pseudo-residual projection coefficients ($$ \beta^t $$).
        """
        self._check_time_access(t)
        G = torch.stack(self.residual[:t], dim=1)
        g = self.residual[t]
        return compute_projection_coef_from(G, g)
    
    def compute_forward_noise_variance(self, t: int) -> float:
        r"""
        Computes the forward noise variance ($$ \vartheta^{[w]} $$).
        """
        self._check_time_access(t)
        if t == 0: # Handles the initial step t=0
            v = self.weight[t]
            res = v
        else: # Handles t>0
            v = self.weight[t] - self.weight[t-1]
            W = torch.stack(self.weight[:t], dim=1)
            alpha = self.compute_forward_projection_coef(t)
            res = v - W @ alpha
    
        return torch.square(res).mean().detach()

    def compute_backward_noise_variance(self, t: int) -> float:
        r"""
        Computes the backward noise variance ($$ \vartheta^{[g]} $$).
        """
        self._check_time_access(t)
        g = self.residual[t]
        if t == 0: # Handles the initial step t=0
            res = g 
        else: # Handles t>0
            G = torch.stack(self.residual[:t], dim=1)
            beta = self.compute_residual_projection_coef(t)
            res = g - G @ beta
            
        return torch.square(res).mean().detach()

    def compute_forward_noise(self, t: int) -> torch.Tensor:
        r"""
        Computes the forward gaussian noise ($$ \omega^{t} $$).
        """
        self._check_time_access(t)

        # Turn off gradients for the statistical sampler
        with torch.no_grad():
            variance = self.compute_forward_noise_variance(t)
            innovation = (variance ** 0.5) * torch.randn(self.K, dtype=torch.float64, requires_grad=True)

            if t == 0: # Handles the initial step t=0
                omega = innovation
            else: # Handles t>0
                Omega = torch.stack(self.forward_noise[:t], dim=1)
                alpha = self.compute_forward_projection_coef(t)
                omega = Omega @ alpha + self.forward_noise[t-1] + innovation

        # Make the sampled noise a base leaf-node for Autograd
        omega = omega.detach().requires_grad_()
        self.forward_noise[t] = omega

        return omega

    def compute_backward_noise(self, t: int) -> torch.Tensor:
        r"""
        Computes the backward gaussian noise ($$ \xi^{t} $$).
        """
        self._check_time_access(t)

        # Turn off gradients for the statistical sampler
        with torch.no_grad():
            variance = self.compute_backward_noise_variance(t)
            innovation = (variance ** 0.5) * torch.randn(self.K, dtype=torch.float64, requires_grad=True)

            if t == 0: # Handles the initial step t=0
                xi = innovation
            else: # Handles t>0
                Xi = torch.stack(self.backward_noise[:t], dim=1)
                beta = self.compute_residual_projection_coef(t)
                xi = Xi @ beta + innovation
        
        # Make the sampled noise a base leaf-node for Autograd
        xi = xi.detach().requires_grad_()
        self.backward_noise[t] = xi

        return xi 

    def compute_selection_rate(self, t: int) -> torch.Tensor:
        """
        Computes the unlabeled-conditional selection rate ($$ A_t $$).

        The Monte Carlo estimate is a deterministic state parameter in the
        scalar effective process. It is therefore detached before it is reused
        in the differentiable pseudo-residual trajectory.
        """
        self._check_time_access(t)
        if self.preactivation[t] is None:
            A = self.indicator.new_zeros(())
        else:
            preactivation = self.preactivation[t]
            mask = self.algo_cfg.selection_function(
                preactivation, 
                self.algo_cfg.positive_margin, 
                self.algo_cfg.negative_margin
            )
            unlabeled = 1.0 - self.indicator
            denominator = torch.sum(unlabeled)
            if denominator.item() == 0.0:
                A = preactivation.new_zeros(())
            else:
                A = (torch.sum(unlabeled * mask) / denominator).detach()

        A = A.detach()
    
        self.selection_rate[t] = A

        return A
