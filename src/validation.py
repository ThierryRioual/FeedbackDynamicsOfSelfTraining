from typing import Optional 
import numpy as np


def validate_self_training_data(
        X_lab: np.ndarray, Y_lab: np.ndarray, 
        X_unl: np.ndarray, 
        X_test: np.ndarray = None, Y_test: np.ndarray = None
        ) -> tuple[int, int, int, Optional[int]]:
    """
    Validate the shapes, types, and finiteness of labeled, unlabeled, and test arrays 
    and return the key counts (N, M, N_test, d).
    """
    assert isinstance(X_lab, np.ndarray), "X_lab must be a numpy array."
    assert isinstance(X_unl, np.ndarray), "X_unl must be a numpy array."
    assert isinstance(Y_lab, np.ndarray), "Y_lab must be a numpy array."

    assert np.all(np.isfinite(X_lab)), "X_lab contains NaN or Inf values."
    assert np.all(np.isfinite(X_unl)), "X_unl contains NaN or Inf values."
    assert np.all(np.isfinite(Y_lab)), "Y_lab contains NaN or Inf values."

    assert X_lab.dtype == np.float64, f"X_lab must be float64, got {X_lab.dtype}."
    assert X_unl.dtype == np.float64, f"X_unl must be float64, got {X_unl.dtype}."
    assert Y_lab.dtype == np.float64, f"Y_lab must be float64, got {Y_lab.dtype}."

    assert X_lab.ndim == 2, "X_lab must be a 2D array."
    assert X_unl.ndim == 2, "X_unl must be a 2D array."
    assert Y_lab.ndim == 1, "Y_lab must be a 1D array."

    assert X_lab.shape[1] == X_unl.shape[1], "X_lab and X_unl must have the same number of features."
    assert X_lab.shape[0] > 0, "X_lab must contain at least one labeled sample."
    assert X_unl.shape[0] >= 0, "X_unl must contain a non-negative number of samples."
    assert Y_lab.shape[0] == X_lab.shape[0], "Number of labeled samples must match number of labels."

    N, d = X_lab.shape
    M = X_unl.shape[0]

    assert (X_test is None) == (Y_test is None), "X_test and Y_test must be provided together."
    assert X_test is None or isinstance(X_test, np.ndarray), "X_test must be a numpy array when provided."
    assert Y_test is None or isinstance(Y_test, np.ndarray), "Y_test must be a numpy array when provided."
    
    if X_test is not None:
        assert X_test.dtype == np.float64, f"X_test must be float64, got {X_test.dtype}."
        assert np.all(np.isfinite(X_test)), "X_test contains NaN or Inf values from DGP."
        assert X_test.ndim == 2, "X_test must be a 2D array when provided."
        assert Y_test.ndim == 1, "Y_test must be a 1D array when provided."
        assert X_test.shape[1] == d, "X_test must have the same number of features as X_lab and X_unl."
        assert Y_test.shape[0] == X_test.shape[0], "Number of test samples must match number of test labels."

    N_test = None if X_test is None else X_test.shape[0]
    return N, M, d, N_test

def validate_dgp_parameters(
        d: int, s: int, 
        spikes_val: np.ndarray, spikes_vect: np.ndarray, 
        mu: np.ndarray, sigma: float, p: float
        ) -> np.ndarray:
    """
    Validate the parameters of the spiked isotropic distribution.
    Returns the validated spike matrix of shape (d, s).
    """
    assert isinstance(d, (int, np.integer)) and d > 0, "Dimension d must be a positive integer."
    assert isinstance(s, (int, np.integer)) and 0 <= s <= d, "Number of spikes s must be a non-negative integer less than or equal to d."

    spikes_val_arr = np.atleast_1d(np.asarray(spikes_val, dtype=float))
    assert spikes_val_arr.size == s, f"spikes_val must contain exactly {s} entries."
    assert s == 0 or spikes_val_arr.ndim == 1, "spikes_val must be a 1D array-like." 
    assert s == 0 or np.all(spikes_val_arr > 0), "All spike variances must be positive."

    assert spikes_vect is None or np.asarray(spikes_vect).ndim == 2, "spikes_vect must be a 2D array when provided."
    V = np.empty((d, 0), dtype=float) if spikes_vect is None else np.asarray(spikes_vect, dtype=float)
    assert spikes_vect is None or V.shape == (d, s), f"spikes_vect must have shape ({d}, {s})."
    assert s == 0 or spikes_vect is not None, "spikes_vect cannot be None when s > 0."

    mu_arr = np.asarray(mu, dtype=float)
    assert mu_arr.ndim == 1 and mu_arr.shape == (d,), f"mu must be a 1D array of shape ({d},)."

    assert isinstance(sigma, (int, float, np.integer, np.floating)) and not np.isnan(sigma) and sigma >= 0, "sigma must be a non-negative real number."
    assert isinstance(p, (int, float, np.integer, np.floating)) and not np.isnan(p) and 0.0 <= p <= 1.0, "p must be a real number between 0.0 and 1.0."

    assert s == 0 or np.allclose(V.T @ V, np.eye(s), atol=1e-8), "The spike vectors must be orthonormal."

    return V

def validate_gradient_step(t: int, weights: np.ndarray, grad: np.ndarray) -> None:
    """
    Validates numerical integrity during the optimization step.
    """
    if not np.all(np.isfinite(grad)):
        bias_weight = weights[0]
        feature_weights_max = np.nanmax(np.abs(weights[1:])) if len(weights) > 1 else np.nan
        grad_bias = grad[0]
        grad_features_max = np.nanmax(np.abs(grad[1:])) if len(grad) > 1 else np.nan
        
        raise ValueError(
            f"Gradient divergence at iteration {t}.\n"
            f"w_0: {bias_weight}, Max w_j: {feature_weights_max}\n"
            f"grad_0: {grad_bias}, Max grad_j: {grad_features_max}"
        )
