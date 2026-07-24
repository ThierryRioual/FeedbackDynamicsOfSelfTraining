import torch
import math
from typing import Optional

def validate_self_training_data(
        X_lab: torch.Tensor, Y_lab: torch.Tensor, 
        X_unl: torch.Tensor, 
        X_test: Optional[torch.Tensor] = None, Y_test: Optional[torch.Tensor] = None
        ) -> tuple[int, int, int, Optional[int]]:
    """
    Validate the shapes, types, and finiteness of labeled, unlabeled, and test tensors 
    and return the key counts (N, M, N_test, d).
    """
    assert isinstance(X_lab, torch.Tensor), "X_lab must be a PyTorch tensor."
    assert isinstance(X_unl, torch.Tensor), "X_unl must be a PyTorch tensor."
    assert isinstance(Y_lab, torch.Tensor), "Y_lab must be a PyTorch tensor."

    assert torch.isfinite(X_lab).all(), "X_lab contains NaN or Inf values."
    assert torch.isfinite(X_unl).all(), "X_unl contains NaN or Inf values."
    assert torch.isfinite(Y_lab).all(), "Y_lab contains NaN or Inf values."

    assert X_lab.dtype == torch.float64, f"X_lab must be float64, got {X_lab.dtype}."
    assert X_unl.dtype == torch.float64, f"X_unl must be float64, got {X_unl.dtype}."
    assert Y_lab.dtype in (torch.float64, torch.int64), f"Y_lab must be float64 or int64, got {Y_lab.dtype}."

    assert X_lab.ndim == 2, "X_lab must be a 2D tensor."
    assert X_unl.ndim == 2, "X_unl must be a 2D tensor."
    assert Y_lab.ndim == 1, "Y_lab must be a 1D tensor."

    assert X_lab.shape[1] == X_unl.shape[1], "X_lab and X_unl must have the same number of features."
    assert X_lab.shape[0] > 0, "X_lab must contain at least one labeled sample."
    assert X_unl.shape[0] >= 0, "X_unl must contain a non-negative number of samples."
    assert Y_lab.shape[0] == X_lab.shape[0], "Number of labeled samples must match number of labels."

    N, d = X_lab.shape
    M = X_unl.shape[0]

    assert (X_test is None) == (Y_test is None), "X_test and Y_test must be provided together."
    assert X_test is None or isinstance(X_test, torch.Tensor), "X_test must be a PyTorch tensor when provided."
    assert Y_test is None or isinstance(Y_test, torch.Tensor), "Y_test must be a PyTorch tensor when provided."
    
    if X_test is not None:
        assert X_test.dtype == torch.float64, f"X_test must be float64, got {X_test.dtype}."
        assert torch.isfinite(X_test).all(), "X_test contains NaN or Inf values from DGP."
        assert X_test.ndim == 2, "X_test must be a 2D tensor when provided."
        assert Y_test.ndim == 1, "Y_test must be a 1D tensor when provided."
        assert X_test.shape[1] == d, "X_test must have the same number of features as X_lab and X_unl."
        assert Y_test.shape[0] == X_test.shape[0], "Number of test samples must match number of test labels."

    N_test = None if X_test is None else X_test.shape[0]
    return N, M, d, N_test

def validate_dgp_parameters(
        d: int, s: int, 
        spikes_val: torch.Tensor, spikes_vect: Optional[torch.Tensor], 
        mu: torch.Tensor, sigma: float, p: float
        ) -> torch.Tensor:
    """
    Validate the parameters of the spiked isotropic distribution.
    Returns the validated spike matrix of shape (d, s).
    """
    assert isinstance(d, int) and d > 0, "Dimension d must be a positive integer."
    assert isinstance(s, int) and 0 <= s <= d, "Number of spikes s must be a non-negative integer less than or equal to d."

    spikes_val_t = torch.atleast_1d(torch.as_tensor(spikes_val, dtype=torch.float64))
    assert spikes_val_t.numel() == s, f"spikes_val must contain exactly {s} entries."
    assert s == 0 or spikes_val_t.ndim == 1, "spikes_val must be a 1D tensor." 
    assert s == 0 or torch.all(spikes_val_t > 0), "All spike variances must be positive."

    assert spikes_vect is None or torch.as_tensor(spikes_vect).ndim == 2, "spikes_vect must be a 2D tensor when provided."
    V = torch.empty((d, 0), dtype=torch.float64) if spikes_vect is None else torch.as_tensor(spikes_vect, dtype=torch.float64)
    assert spikes_vect is None or V.shape == (d, s), f"spikes_vect must have shape ({d}, {s})."
    assert s == 0 or spikes_vect is not None, "spikes_vect cannot be None when s > 0."

    mu_t = torch.as_tensor(mu, dtype=torch.float64)
    assert mu_t.ndim == 1 and mu_t.shape == (d,), f"mu must be a 1D tensor of shape ({d},)."

    assert isinstance(sigma, (int, float)) and not math.isnan(sigma) and sigma >= 0, "sigma must be a non-negative real number."
    assert isinstance(p, (int, float)) and not math.isnan(p) and 0.0 <= p <= 1.0, "p must be a real number between 0.0 and 1.0."

    assert s == 0 or torch.allclose(V.T @ V, torch.eye(s, dtype=torch.float64), atol=1e-8), "The spike vectors must be orthonormal."

    return V

def validate_gradient_step(t: int, weights: torch.Tensor, grad: torch.Tensor) -> None:
    """
    Validates numerical integrity during the optimization step.
    """
    if not torch.isfinite(grad).all():
        bias_weight = weights[0].item()
        
        # PyTorch equivalent of np.nanmax(np.abs(...))
        feature_weights_max = torch.nanmax(torch.abs(weights[1:])).item() if len(weights) > 1 else float('nan')
        grad_bias = grad[0].item()
        grad_features_max = torch.nanmax(torch.abs(grad[1:])).item() if len(grad) > 1 else float('nan')
        
        raise ValueError(
            f"Gradient divergence at iteration {t}.\n"
            f"w_0: {bias_weight}, Max w_j: {feature_weights_max}\n"
            f"grad_0: {grad_bias}, Max grad_j: {grad_features_max}"
        )