import torch

from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, List, Dict, Set

from src.config import DataConfig, AlgorithmConfig
from src.objectives import LossFunction, Penalty, LogisticLoss, RidgePenalty



@dataclass
class MacroscopicStateEvolution:
    """
    Numerically integrate the state evolution equations and compute the deterministic theoretical limits using Monte Carlo averaging.
    """
    # Frozen configuration objects (same pattern as SelfTrainedGradientDescent / IsotropicGaussian)
    data_cfg: DataConfig
    algo_cfg: AlgorithmConfig

    mc_seed: int = field(default=42) # monte carlo seed
    K: int = field(default=1000) # number of monte carlo samples

    initial_bias: torch.float64 = field(default=0.0) # b_{init}
    initial_weight: torch.Tensor = field(default=None) # w_{init}

    # Instance variables    
    signal: torch.Tensor = field(init=False) # \mu
    label: torch.Tensor = field(init=False) # Y
    indicator: torch.Tensor = field(init=False) # \Delta

    bias: List[torch.Tensor] = field(init=False) # b 
    weight: List[torch.Tensor] = field(init=False) # w

    loc_field: List[torch.Tensor] = field(init=False) # r (local field)
    residual: List[torch.Tensor] = field(init=False) # g (pseudo-residual)

    weight_memory: List[torch.Tensor] = field(init=False) # \gamma^{[w]}
    residual_memory: List[torch.Tensor] = field(init=False) # \gamma^{[g]}

    weight_signal_alignments: List[torch.Tensor] = field(init=False) # m
    label_residual_cov: List[torch.Tensor] = field(init=False) # \chi
    mean_residual: List[torch.Tensor] = field(init=False) # \zeta
    selection_rate: List[torch.Tensor] = field(init=False) # A


    def __post_init__(self):
        """
        Initialize the state evolution.
        """
        torch.manual_seed(self.mc_seed)

        self.signal = torch.tensor([self.data_cfg.signal_prior() for _ in range(self.K)]) # signal vector
        self.label = (torch.rand(self.K) < self.data_cfg.label_prior).double() * 2 - 1 # transform to {+1, -1}
        self.indicator = (torch.rand(self.K) < self.data_cfg.observation_prior).double() # observation indicator 

        self.initial_bias = torch.tensor(self.initial_bias, requires_grad=True)

        if self.initial_weight is None:
            self.initial_weight = torch.randn(self.K, requires_grad=True)
        else:
            self.initial_weight = torch.tensor(self.initial_weight, requires_grad=True)

        T = self.algo_cfg.n_iterations

        self.bias = [None] * (T + 1)
        self.weight = [None] * (T + 1)

        self.bias[0] = self.initial_bias
        self.weight[0] = self.initial_weight

        self.loc_field = [None] * T
        self.residual = [None] * T

        self.forward_noise = [None] * T
        self.backward_noise = [None] * T

        self.weight_memory = [None] * T
        self.residual_memory = [None] * T

        self.weight_signal_alignments = [None] * T
        self.label_residual_cov = [None] * T
        self.mean_residual = [None] * T

        self.selection_rate = [None] * T

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
        return torch.tensor(self.data_cfg.observation_prior)
    
    @property
    def delta(self) -> float:
        """
        Returns the data-to-dimension ratio.
        """
        return torch.tensor(self.data_cfg.data_to_dimension_ratio)

    @property
    def eta(self) -> float:
        """
        Returns the learning rate.
        """
        return torch.tensor(self.algo_cfg.step_size)

    def compute_decay(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Returns the weight decay (penalty) gradient function: h = -η λ ∇P(w).
        """
        return - self.eta * self.algo_cfg.penalty_param * self.algo_cfg.penalty_function.gradient(weight)
    
    def compute_residual(self, loc_field: torch.Tensor, 
            label: torch.Tensor, indicator: torch.Tensor,
            coef: float, selection_rate: float,
        ) -> torch.Tensor:
        """
        Computes the pseudo-residual g = -η ∇_r R(r).
        This is the gradient of the empirical risk with respect to the field r, not the weights.
        """
        rho = self.data_cfg.observation_prior
        kappa = self.algo_cfg.margin_threshold
        labeled_grad = self.algo_cfg.loss_function.gradient(loc_field, label)
        unlabeled_grad = self.algo_cfg.loss_function.gradient(loc_field, torch.where(loc_field >= 0, 1, -1))
        selection_mask = torch.where(torch.abs(loc_field) >= kappa, 1.0, 0.0)
        if coef == 0.0: # allows for selection rate of 0.0 at initialization
            return - self.eta * (indicator / rho) * labeled_grad
        else:
            return - self.eta * ((indicator / rho) * labeled_grad + coef * ((1 - indicator) / (1 - rho)) * unlabeled_grad * (selection_mask / selection_rate ))
    
    def _check_time_access(self, t: int) -> None:
        """
        Ensures that helper methods do not access uncomputed future states.
        """
        if t > self._current_t:
            raise ValueError(f"Cannot access uncomputed future steps. Current t is {self._current_t}.")

    def compute_trajectory(self):
        """Computes the full macroscopic trajectory of the algorithm."""
        for t in range(self.T):
            self.step(t)

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
        b = self.bias[t] # b_{t}
        pi = self.algo_cfg.get_pseudo_label_weight(t)

        if t == 0: # Handles initial step t=0 (Empty sum)
            r = b * torch.ones(self.K) + m * self.label + omega

        else: # Handles t>0
            gamma = self.compute_weight_memory(t) # \gamma^{[w]}
            G = torch.stack(self.residual[:t], dim=1) # G_{t-1} (residual matrix)
            r = b * torch.ones(self.K) + m * self.label + G @ gamma + omega # r^{t}

        self.loc_field[t] = r # r (local field)

        A = self.compute_selection_rate(t)
        g = self.compute_residual(r, self.label, self.indicator, pi, A) # g^{t}
        self.residual[t] = g # g (pseudo-residual)

        return r, g

    def backward_pass(self, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the backward pass of the self-training algorithm.
        Computes the updated bias and weight using the pseudo-residual memory and the decay step.
        """

        gamma = self.compute_residual_memory(t) # \gamma^{[g]}
        chi = self.compute_label_residual_cov(t) # \chi
        zeta = self.compute_mean_residual(t) # \zeta
        xi = self.compute_backward_noise(t) # \xi

        prev_b = self.bias[t]
        prev_w = self.weight[t]

        W = torch.stack(self.residual[:t+1], dim=1) # W_{t} (residual matrix)
        h = self.compute_decay(prev_w) # h (weight decay)

        b = prev_b + zeta
        self.bias[t+1] = b

        w = prev_w + h + chi * self.signal + (W @ gamma + xi) / torch.sqrt(self.delta)
        self.weight[t+1] = w

        return b, w

    @property
    def weight_grammian(self) -> torch.Tensor:
        """
        Computes the weight Gram matrix ($$ C^{[w]} $$)
        """
        W = torch.stack(self.weight[:self._current_t], dim=1)
        return W.T @ W / self.K
    
    @property
    def residual_grammian(self) -> torch.Tensor:
        """
        Computes the pseudo-residual Gram matrix ($$ C^{[g]} $$)
        """
        G = torch.stack(self.residual[:self._current_t], dim=1)
        return G.T @ G / self.K
    
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
            retain_graph=True
        )

        gamma = torch.tensor([torch.mean(dw).item() for dw in derivatives])
        self.weight_memory[t] = gamma

        return gamma

    def compute_residual_memory(self, t: int) -> torch.Tensor:
        """
        Computes the pseudo-residual memory vector ($$ \gamma^{[g]} $$)
        """
        self._check_time_access(t)
        current_residual = self.residual[t] # g^t
        noise_inputs = tuple(self.forward_noise[:t+1]) # Pass sequence of original tensors

        derivatives = torch.autograd.grad(
            outputs=current_residual,
            inputs=noise_inputs,
            grad_outputs=torch.ones_like(current_residual),
            retain_graph=True
        )

        gamma = torch.tensor([torch.mean(dg).item() for dg in derivatives])
        self.residual_memory[t] = gamma

        return gamma

    def compute_weight_signal_alignment(self, t: int) -> torch.Tensor:
        """
        Computes the weight-signal alignment ($$ m $$)
        """
        self._check_time_access(t)
        w = self.weight[t] # w^t
        mu = self.signal
        m = torch.dot(w, mu) / self.K
        self.weight_signal_alignments[t] = m
        return m

    def compute_label_residual_cov(self, t: int) -> torch.Tensor:
        """
        Computes the label-residual covariance ($$ \chi $$)
        """
        self._check_time_access(t)
        g = self.residual[t] # g^t
        y = self.label
        chi = torch.dot(g, y) / self.K
        self.label_residual_cov[t] = chi
        return chi

    def compute_mean_residual(self, t: int) -> torch.Tensor:
        """
        Computes the mean pseudo-residual ($$ \zeta $$)
        """
        self._check_time_access(t)
        g = self.residual[t] # g^t
        ones_vec = torch.ones(self.K)
        zeta = torch.dot(g, ones_vec) / self.K
        self.mean_residual[t] = zeta
        return zeta    

    @staticmethod
    def compute_projection_coef(X: torch.Tensor, x: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
        """
        Computes the projection coefficients of x onto the span of X.
        Uses Tikhonov-regularized normal equations to guarantee autograd 
        stability during State Evolution.
        """
        # 1. Compute the Gram matrix (X^T X)
        # If X is (K, t), gram_matrix is a small (t, t) matrix.
        gram_matrix = X.T @ X
        
        # 2. Inject diagonal jitter for numerical stability
        gram_matrix += jitter * torch.eye(gram_matrix.shape[0], device=X.device, dtype=X.dtype)
        
        # 3. Compute the right-hand side (X^T x)
        rhs = X.T @ x
        
        # 4. Solve the well-conditioned system (highly stable autograd)
        return torch.linalg.solve(gram_matrix, rhs)

    def compute_forward_projection_coef(self, t: int) -> torch.Tensor:
        r"""
        Computes the forward projection coefficients ($$ \alpha^t $$).
        """
        self._check_time_access(t)
        W = torch.stack(self.weight[:t], dim=1)
        v = self.weight[t] - self.weight[t-1] if t > 0 else self.weight[t]
        return self.compute_projection_coef(W, v)

    def compute_residual_projection_coef(self, t: int) -> torch.Tensor:
        r"""
        Computes the pseudo-residual projection coefficients ($$ \beta^t $$).
        """
        self._check_time_access(t)
        G = torch.stack(self.residual[:t], dim=1)
        g = self.residual[t]
        return self.compute_projection_coef(G, g)
    
    def compute_forward_noise_variance(self, t: int) -> torch.Tensor:
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
    
        return torch.square(res).mean()

    def compute_backward_noise_variance(self, t: int) -> torch.Tensor:
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
            
        return torch.square(res).mean()

    def compute_forward_noise(self, t: int) -> torch.Tensor:
        r"""
        Computes the forward gaussian noise ($$ \omega^{t} $$).
        """
        self._check_time_access(t)
        variance = self.compute_forward_noise_variance(t)
        innovation = torch.sqrt(variance) * torch.randn(self.K)

        if t == 0: # Handles the initial step t=0
            omega = innovation
        else: # Handles t>0
            Omega = torch.stack(self.forward_noise[:t], dim=1)
            alpha = self.compute_forward_projection_coef(t)
            omega = Omega @ alpha + self.forward_noise[t-1] + innovation

        self.forward_noise[t] = omega

        return omega

    def compute_backward_noise(self, t: int) -> torch.Tensor:
        r"""
        Computes the backward gaussian noise ($$ \xi^{t} $$).
        """
        self._check_time_access(t)
        variance = self.compute_backward_noise_variance(t)
        innovation = torch.sqrt(variance) * torch.randn(self.K)

        if t == 0: # Handles the initial step t=0
            xi = innovation
        else: # Handles t>0
            Xi = torch.stack(self.backward_noise[:t], dim=1)
            beta = self.compute_residual_projection_coef(t)
            xi = Xi @ beta + innovation
        
        self.backward_noise[t] = xi

        return xi 

    def compute_selection_rate(self, t: int) -> torch.Tensor:
        """
        Computes the selection rate ($$ A_t $$)
        """
        self._check_time_access(t)
        if self.loc_field[t] is None:
            A = torch.tensor(0.0)
        else:
            loc_field = self.loc_field[t]
            A = torch.where(torch.abs(loc_field) >= self.algo_cfg.margin_threshold, 1.0, 0.0).mean()
    
        self.selection_rate[t] = A

        return A