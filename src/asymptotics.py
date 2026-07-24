import torch

from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, List, Dict, Set


@dataclass
class MacroscopicStateEvolution:
    """
    Numerically integrate the state evolution equations and compute the deterministic theoretical limits using Monte Carlo averaging.
    """
    # Initialization paramters

    sample_to_dimension_ratio: float
    signal_prior: Callable[[int], torch.Tensor]
    label_prior: float 

    score_function: Callable[[torch.Tensor], torch.Tensor]
    penalty_gradient: Callable[[torch.Tensor], torch.Tensor]

    mc_seed: int = field(default=42) # monte carlo seed
    K: int = field(default=1000) # number of monte carlo samples

    inital_bias: float = field(default=0) # b_{init}
    initial_weight: torch.Tensor = field(default=None) # w_{init}

    # Instance variables

    signal: torch.Tensor = field(init=False)
    label: torch.Tensor = field(init=False)

    bias: List[torch.Tensor] = field(init=False) # b 
    weight: List[torch.Tensor] = field(init=False) # w

    prediction: List[torch.Tensor] = field(init=False) # r
    score: List[torch.Tensor] = field(init=False) # g

    weight_memory: List[torch.Tensor] = field(init=False) # \gamma^{[w]}
    score_memory: List[torch.Tensor] = field(init=False) # \gamma^{[g]}

    weight_signal_alignments: List[torch.Tensor] = field(init=False) # m
    score_label_alignments: List[torch.Tensor] = field(init=False) # \chi
    expected_score: List[torch.Tensor] = field(init=False) # \zeta

    def __post_init__(self):
        """
        Initialize the state evolution.
        """
        torch.manual_seed(self.mc_seed)

        self.signal = self.signal_prior(self.K)
        self.label = (torch.rand(self.K) < self.label_prior).float() * 2 - 1 # transform to {+1, -1}

        self.initial_bias = torch.tensor(self.initial_bias, requires_grad=True)

        if self.initial_weight is None:
            self.initial_weight = torch.randn(self.K, requires_grad=True)
        else:
            self.initial_weight = torch.tensor(self.initial_weight, requires_grad=True)

        self.bias = [self.initial_bias]
        self.weight = [self.initial_weight]

        self.prediction = []
        self.score = []

        self.forward_noise = []
        self.backward_noise = []

        self.weight_memory = []
        self.score_memory = []

        self.weight_signal_alignments = []
        self.score_label_alignments = []
        self.expected_score = []

    def forward_pass(self):   
        """
        Performs the forward pass of the self-training algorithm.
        """
        gamma = self.compute_weight_memory() # \gamma^{[w]}
        m = self.compute_weight_signal_alignment() # m
        omega = self.compute_forward_noise() # \omega

        G = torch.stack(self.score[:-1], dim=1) # G_{t-1}
        b = self.bias[-1] # b_{t}

        r = b * torch.ones(self.K) + m * self.label + G @ gamma + omega # r^{t}
        g = self.score_function(r) # g^{t}

        self.prediction.append(r) # r
        self.score.append(g) # g

        return r, g

    def backward_pass(self):
        """
        Performs the backward pass of the self-training algorithm.
        """
        gamma = self.compute_score_memory() # \gamma^{[g]}
        chi = self.compute_score_label_alignment() # \chi
        zeta = self.compute_expected_score() # \zeta
        xi = self.compute_backward_noise() # \xi

        prev_b = self.bias[-1]
        prev_w = self.weight[-1]

        W = torch.stack(self.score, dim=1) # W_{t}
        h = self.penalty_gradient(prev_w) # h

        b = prev_b + zeta # g^{t}
        w = prev_w + h + chi * self.signal + W @ gamma + xi # r^{t}

        self.bias.append(b) # g
        self.weight.append(w) # r

        return b, w

    @property
    def weight_grammian(self) -> torch.Tensor:
        """
        Computes the weight Gram matrix ($$ C^{[w]} $$)
        """
        W = torch.stack(self.weight, dim=1)
        return W.T @ W / self.K
    
    @property
    def score_grammian(self) -> torch.Tensor:
        """
        Computes the score Gram matrix ($$ C^{[g]} $$)
        """
        G = torch.stack(self.score, dim=1)
        return G.T @ G / self.K
    
    def compute_weight_memory(self) ->torch.Tensor:
        """
        Computes the weight memory vector ($$ \gamma^{[w]} $$)
        """
        current_weight = self.weight[-1] # w^t
        backward_noise_matrix = torch.stack(self.backward_noise[:-1], dim=0) # \Xi_{t-1}

        derivatives = torch.autograd.grad(
            outputs=current_weight,
            inputs=backward_noise_matrix,
            grad_outputs=torch.ones_like(current_weight),
            retain_graph=True
        )

        gamma = torch.tensor([torch.mean(dw).item() for dw in derivatives])
        self.weight_memory.append(gamma)

        return gamma

    def compute_score_memory(self) -> torch.Tensor:
        """
        Computes the score memory vector ($$ \gamma^{[g]} $$)
        """
        current_score = self.score[-1] # g^t
        forward_noise_matrix = torch.stack(self.forward_noise, dim=0) # \Omega_{t}

        derivatives = torch.autograd.grad(
            outputs=current_score,
            inputs=forward_noise_matrix,
            grad_outputs=torch.ones_like(current_score),
            retain_graph=True
        )

        gamma = torch.tensor([torch.mean(dg).item() for dg in derivatives])
        self.score_memory.append(gamma)

        return gamma

    def compute_weight_signal_alignment(self) -> torch.Tensor:
        """
        Computes the weight-signal alignment vector ($$ m $$)
        """
        w = self.weight[-1] # w^t
        mu = self.signal
        m = torch.dot(w, mu) / self.K
        self.weight_signal_alignments.append(m)
        return m

    def compute_score_label_alignment(self) -> torch.Tensor:
        """
        Computes the score-label alignment vector ($$ \chi $$)
        """
        g = self.score[-1] # g^t
        y = self.label
        chi = torch.dot(g, y) / self.K
        self.score_label_alignments.append(chi)
        return chi

    def compute_expected_score(self) -> torch.Tensor:
        """
        Computes the expected scores ($$ \zeta $$)
        """
        g = self.score[-1] # g^t
        ones_vec = torch.ones(self.K)
        zeta = torch.dot(g, ones_vec) / self.K
        self.expected_score.append(zeta)
        return zeta    

    @staticmethod
    def _compute_projection_coef(X: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the projection coefficients of x onto the span of X.
        ($$ X^\dagger x $$)
        """
        return torch.linalg.lstsq(X, x).solution

    def compute_forward_projection_coef(self) -> torch.Tensor:
        r"""
        Computes the forward projection coefficients ($$ \alpha^t $$).
        """
        W = torch.stack(self.weight[:-1], dim=1)
        v = self.weight[-1] - self.weight[-2] if len(self.weight) > 1 else self.weight[-1]
        return self._compute_projection_coef(W, v)

    def compute_backward_projection_coef(self) -> torch.Tensor:
        r"""
        Computes the backward projection coefficients ($$ \beta^t $$).
        """
        G = torch.stack(self.score[:-1], dim=1)
        g = self.score[-1]
        return self._compute_projection_coef(G, g)
    
    def compute_forward_noise_variance(self) -> torch.Tensor:
        r"""
        Computes the forward noise variance ($$ \vartheta^{[w]} $$).
        """
        W = torch.stack(self.weight[:-1], dim=1)
        v = self.weight[-1] - self.weight[-2] if len(self.weight) > 1 else self.weight[-1]
        alpha = self.compute_forward_projection_coef()
        residual = v - W @ alpha
        return torch.square(residual).mean()

    def compute_backward_noise_variance(self) -> torch.Tensor:
        r"""
        Computes the backward noise variance ($$ \vartheta^{[g]} $$).
        """
        G = torch.stack(self.score[:-1], dim=1)
        g = self.score[-1]
        beta = self.compute_backward_projection_coef()
        residual = g - G @ beta
        return torch.square(residual).mean()

    def compute_forward_noise(self) -> torch.Tensor:
        r"""
        Computes the forward gaussian noise ($$ \omega^{t} $$).
        """
        variance = self.compute_forward_noise_variance()
        innovation = torch.sqrt(variance) * torch.randn(self.K, device=self.device, dtype=self.dtype)
        
        Omega = torch.stack(self.forward_noise[:-1], dim=0)
        alpha = self.compute_forward_projection_coef()

        omega = Omega @ alpha + self.forward_noise[:-1] + innovation
        self.forward_noise.append(omega)

        return omega

    def compute_backward_noise(self) -> torch.Tensor:
        r"""
        Computes the backward gaussian noise ($$ \xi^{t} $$).
        """
        variance = self.compute_backward_noise_variance()
        innovation = torch.sqrt(variance) * torch.randn(self.K, device=self.device, dtype=self.dtype)
        
        Xi = torch.stack(self.backward_noise[:-1], dim=0)
        beta = self.compute_backward_projection_coef()

        xi = Xi @ beta + innovation
        self.backward_noise.append(xi)

        return xi 

        

            