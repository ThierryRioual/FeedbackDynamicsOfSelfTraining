from __future__ import annotations

import math
import re
import warnings
from typing import Optional, TYPE_CHECKING

import torch
from src.performance import population_error
from src.primitives import pseudo_labels, pseudo_residual

if TYPE_CHECKING:
    from src.asymptotics import MacroscopicStateEvolution

def compute_projection_coef_from(X: torch.Tensor, x: torch.Tensor, rcond: Optional[float] = None) -> torch.Tensor:
    """Return least-squares coefficients without forming normal equations.

    This generic legacy helper is not used by the production state evolution,
    which works in incremental orthogonal trajectory coordinates.
    """
    solution = torch.linalg.lstsq(X, x, rcond=rcond).solution
    return solution.detach()

def compute_abstract_pseudo_residual_from(
    preactivation: torch.Tensor,
    label: torch.Tensor,
    indicator: torch.Tensor,
    selection_mask: torch.Tensor,
    selection_rate: float,
    coef: float,
    rho: float,
    eta: float,
    loss_function,  # Type hinted as your base LossFunction abstract class
    time_index: Optional[int] = None,
    initial_pseudo_label: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Core mathematical engine for computing the pseudo-residual g = -η ∇_r R(r).
    Shared by theoretical State Evolution and empirical Gradient Descent tracking.

    Existing callers that omit ``time_index`` retain the historical
    ``sign(preactivation)`` pseudo-label rule.  A paper-faithful first update is
    requested with ``time_index=0``; in that case an explicit exogenous
    ``initial_pseudo_label`` is required whenever the unlabeled contribution is
    active.  Later updates use ``sign(preactivation)``.
    """
    if not isinstance(preactivation, torch.Tensor):
        raise TypeError("preactivation must be a torch.Tensor")
    if not isinstance(label, torch.Tensor):
        raise TypeError("label must be a torch.Tensor")
    if not isinstance(indicator, torch.Tensor):
        raise TypeError("indicator must be a torch.Tensor")
    if not isinstance(selection_mask, torch.Tensor):
        raise TypeError("selection_mask must be a torch.Tensor")
    if label.shape != preactivation.shape:
        raise ValueError(
            "label and preactivation must have the same shape, "
            f"got {tuple(label.shape)} and {tuple(preactivation.shape)}"
        )
    if selection_mask.shape != preactivation.shape:
        raise ValueError(
            "selection_mask and preactivation must have the same shape, "
            f"got {tuple(selection_mask.shape)} and {tuple(preactivation.shape)}"
        )
    try:
        indicator_shape = torch.broadcast_shapes(
            indicator.shape, preactivation.shape
        )
    except RuntimeError as error:
        raise ValueError(
            "indicator must be scalar or broadcastable to the preactivation "
            f"shape {tuple(preactivation.shape)}, got {tuple(indicator.shape)}"
        ) from error
    if indicator_shape != preactivation.shape:
        raise ValueError(
            "indicator must broadcast exactly to the preactivation shape, "
            f"got {tuple(indicator_shape)} instead of {tuple(preactivation.shape)}"
        )
    if label.device != preactivation.device:
        raise ValueError("label and preactivation must be on the same device")
    if indicator.device != preactivation.device:
        raise ValueError("indicator and preactivation must be on the same device")
    if selection_mask.device != preactivation.device:
        raise ValueError(
            "selection_mask and preactivation must be on the same device"
        )
    if not torch.all((label == 1) | (label == -1)):
        raise ValueError("label entries must belong to {-1, +1}")
    if time_index is not None:
        if isinstance(time_index, bool) or not isinstance(time_index, int):
            raise TypeError("time_index must be a nonnegative integer or None")
        if time_index < 0:
            raise ValueError("time_index must be nonnegative")

    if isinstance(selection_rate, torch.Tensor):
        rate = selection_rate.detach().to(dtype=preactivation.dtype, device=preactivation.device)
        rate_value = rate.item()
    else:
        rate_value = float(selection_rate)
        rate = preactivation.new_tensor(rate_value)

    if not math.isfinite(rate_value):
        raise FloatingPointError(f"selection_rate must be finite, got {rate_value}")
    if rate_value <= 0.0:
        # The objective adopts the convention 0/0 = 0 when no unlabeled
        # observation is selected, so its pseudo-labeled contribution is zero.
        rate = preactivation.new_zeros(())

    if coef == 0.0 or rho >= 1.0 or rate_value <= 0.0:
        # No pseudo-labelled term is active, hence no exogenous label is
        # mathematically needed at t=0.
        return pseudo_residual(
            scores=preactivation,
            Y=label,
            Delta=indicator.to(dtype=preactivation.dtype),
            Yhat=label,
            selection=torch.zeros_like(selection_mask),
            omega=rate,
            pi=0.0,
            eta=eta,
            rho=rho,
            loss_function=loss_function,
        )

    if time_index == 0:
        if initial_pseudo_label is None:
            raise ValueError(
                "initial_pseudo_label is required for an active unlabeled "
                "contribution at time_index=0"
            )
        if not isinstance(initial_pseudo_label, torch.Tensor):
            raise TypeError("initial_pseudo_label must be a torch.Tensor")
        if initial_pseudo_label.shape != preactivation.shape:
            raise ValueError(
                "initial_pseudo_label and preactivation must have the same "
                f"shape, got {tuple(initial_pseudo_label.shape)} and "
                f"{tuple(preactivation.shape)}"
            )
        if initial_pseudo_label.device != preactivation.device:
            raise ValueError(
                "initial_pseudo_label and preactivation must be on the same device"
            )
        if not torch.all(
            (initial_pseudo_label == 1) | (initial_pseudo_label == -1)
        ):
            raise ValueError(
                "initial_pseudo_label entries must belong to {-1, +1}"
            )
        y_init = initial_pseudo_label.to(dtype=preactivation.dtype)
    else:
        y_init = torch.where(preactivation >= 0, 1.0, -1.0)
    # ``time_index=None`` historically meant endogenous sign scores; map it to
    # a positive time to preserve that compatibility while sharing the single
    # manuscript pseudo-residual implementation.
    effective_time = 1 if time_index is None else time_index
    y_pseudo = pseudo_labels(effective_time, preactivation, y_init)
    return pseudo_residual(
        scores=preactivation,
        Y=label,
        Delta=indicator.to(dtype=preactivation.dtype),
        Yhat=y_pseudo,
        selection=selection_mask,
        omega=rate,
        pi=coef,
        eta=eta,
        rho=rho,
        loss_function=loss_function,
    )

def compute_population_error_from(b: float, m: float, tau: float, sigma: float, p: float) -> float:
    """
    Computes the population error.
    """
    return population_error(b=b, m=m, tau=tau, sigma=sigma, p=p)

def plot_autograd_dag(
    se: MacroscopicStateEvolution, 
    t: int, 
    filename: str = "autograd_graph",
    rankdir: str = 'TB',
    dpi: str = '300'
):
    """Legacy visualizer for a gradient-enabled state-evolution trajectory.

    The production state evolution is deliberately graph-free.  Calling this
    helper on production output therefore emits a warning and can only render
    disconnected tensor leaves; use the orthogonal-basis diagnostics instead.
    """
    # Imported only for this diagnostic; production paths are graph-free and
    # should not require the optional torchviz dependency.
    from torchviz import make_dot

    # 1. Base static leaf nodes
    params = {
        "Y": se.label,
        "mu": se.signal,
        "Delta": se.indicator
    }

    # 2. Systematically extract trajectory up to step t
    # We iterate up to t+1 because the backward pass at step t computes w^{t+1} and b^{t+1}
    for i in range(t + 2): 
        
        # Variables that update in the backward pass (go up to t+1)
        if i < len(se.bias) and isinstance(se.bias[i], torch.Tensor):
            params[f"b^{i}"] = se.bias[i]
        if i < len(se.weight) and isinstance(se.weight[i], torch.Tensor):
            params[f"w^{i}"] = se.weight[i]

        # Variables that update in the forward/memory passes (go up to t)
        if i <= t:
            if i < len(se.forward_noise) and isinstance(se.forward_noise[i], torch.Tensor):
                params[f"q^{i}"] = se.forward_noise[i]
            if i < len(se.backward_noise) and isinstance(se.backward_noise[i], torch.Tensor):
                params[f"p^{i}"] = se.backward_noise[i]

            if i < len(se.weight_memory) and isinstance(se.weight_memory[i], torch.Tensor):
                params[f"phi^[w]_{i}"] = se.weight_memory[i]
            if i < len(se.residual_memory) and isinstance(se.residual_memory[i], torch.Tensor):
                params[f"phi^[g]_{i}"] = se.residual_memory[i]

            if i < len(se.residual) and isinstance(se.residual[i], torch.Tensor):
                params[f"g^{i}"] = se.residual[i]
            if i < len(se.preactivation) and isinstance(se.preactivation[i], torch.Tensor):
                params[f"r^{i}"] = se.preactivation[i]

            if i < len(se.weight_signal_alignments) and isinstance(se.weight_signal_alignments[i], torch.Tensor):
                params[f"m_{i}"] = se.weight_signal_alignments[i]
            if i < len(se.label_residual_alignments) and isinstance(se.label_residual_alignments[i], torch.Tensor):
                params[f"chi_{i}"] = se.label_residual_alignments[i]
            if i < len(se.mean_residual) and isinstance(se.mean_residual[i], torch.Tensor):
                params[f"zeta_{i}"] = se.mean_residual[i]
            if i < len(se.selection_rate) and isinstance(se.selection_rate[i], torch.Tensor):
                params[f"A_{i}"] = se.selection_rate[i]


    # 3. Build the exact output target tuple (Green Boxes) used in your original script
    # We order them descending so the newest nodes sit cleanly in the tuple
    target_tensors = []
    for i in range(t, -1, -1):
        if f"g^{i}" in params: target_tensors.append(params[f"g^{i}"])
        if f"r^{i}" in params: target_tensors.append(params[f"r^{i}"])
    for i in range(t + 1, -1, -1):
        if f"b^{i}" in params: target_tensors.append(params[f"b^{i}"])
        if f"w^{i}" in params: target_tensors.append(params[f"w^{i}"])
        
    outputs = tuple(target_tensors)
    if outputs and not any(tensor.requires_grad for tensor in outputs):
        warnings.warn(
            "The production state evolution is graph-free; plot_autograd_dag "
            "can only render disconnected tensor leaves. Inspect B_w, B_g, "
            "Theta_w, Theta_g, Q_w, P_g, and the rank diagnostics instead.",
            RuntimeWarning,
            stacklevel=2,
        )

    # 4. Build the combined memory ID map for labeling
    id_map = {id(tensor): name for name, tensor in params.items()}
    # Inherit the hidden intermediate terms (like Onsager caches) from your class dictionary
    if hasattr(se, '_debug_id_map') and se._debug_id_map is not None:
        id_map.update(se._debug_id_map)

    # 5. Generate the graph 
    # NOTICE: We pass NO params to make_dot! This avoids the scalar float crash entirely.
    dot = make_dot(outputs, show_attrs=True, show_saved=True)

    # 6. Apply visual styling
    dot.attr(dpi=dpi, rankdir=rankdir, bgcolor='white')
    dot.graph_attr.update(
        nodesep='0.5', 
        ranksep='0.8', 
        fontname='Helvetica',
        fontsize='12',
        size='2,2'
    )

    # 7. Inject the math labels via integer address replacement
    for i, line in enumerate(dot.body):
        for tensor_id, custom_label in id_map.items():
            if str(tensor_id) in line:
                line = re.sub(
                    r'label="([^"]*)"', 
                    lambda m: f'label="{m.group(1).replace("SavedTensor", "").replace("self", "").replace("other", "").strip()}\\n--- {custom_label} ---"', 
                    line
                )
                dot.body[i] = line

    # 8. Render directly to disk
    dot.render(filename, format="svg")
    return dot
