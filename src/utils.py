from __future__ import annotations # Tells the interpreter to defer evaluating all type hints
import math

import torch
from scipy.stats import norm
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.asymptotics import MacroscopicStateEvolution

def compute_projection_coef_from(X: torch.Tensor, x: torch.Tensor, rcond: Optional[float] = None) -> torch.Tensor:
    """
    Computes the projection coefficients of x onto the span of X.
    Uses Tikhonov-regularized normal equations to guarantee autograd 
    stability during State Evolution.
    """
    # torch.linalg.lstsq solves min ||X * alpha - x||^2 directly without forming X.T @ X
    # It returns a named tuple; we extract the .solution attribute.
    # rcond=None automatically cuts off precision errors using machine limits.
    solution = torch.linalg.lstsq(X, x, rcond=rcond).solution
    
    # Detach to keep the solver completely out of the Autograd graph
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
    loss_function  # Type hinted as your base LossFunction abstract class
) -> torch.Tensor:
    """
    Core mathematical engine for computing the pseudo-residual g = -η ∇_r R(r).
    Shared by theoretical State Evolution and empirical Gradient Descent tracking.
    """
    labeled_grad = loss_function.gradient(preactivation, label)
    
    y_pseudo = torch.where(preactivation >= 0, 1.0, -1.0)

    unlabeled_grad = loss_function.gradient(preactivation, y_pseudo)
    labeled_contribution = (indicator / rho) * labeled_grad

    if coef == 0.0 or rho >= 1.0:
        return -eta * labeled_contribution

    if isinstance(selection_rate, torch.Tensor):
        rate_value = selection_rate.detach().item()
    else:
        rate_value = float(selection_rate)

    if not math.isfinite(rate_value):
        raise FloatingPointError(f"selection_rate must be finite, got {rate_value}")
    if rate_value <= 0.0:
        # The objective adopts the convention 0/0 = 0 when no unlabeled
        # observation is selected, so its pseudo-labeled contribution is zero.
        return -eta * labeled_contribution

    unlabeled_contribution = (
        coef
        * ((1.0 - indicator) / (1.0 - rho))
        * unlabeled_grad
        * (selection_mask / selection_rate)
    )
    return -eta * (labeled_contribution + unlabeled_contribution)

def compute_population_error_from(b: float, m: float, tau: float, sigma: float, p: float) -> float:
    """
    Computes the population error.
    """
    err = p * norm.cdf((- b - m) / (tau * sigma)) + (1 - p) * norm.cdf((b - m) / (tau * sigma))
    return err

import torch
import re
from torchviz import make_dot

def plot_autograd_dag(
    se: MacroscopicStateEvolution, 
    t: int, 
    filename: str = "autograd_graph",
    rankdir: str = 'TB',
    dpi: str = '300'
):
    """
    Master plotting function to visualize the State Evolution Autograd DAG.
    Automatically extracts history up to step t and annotates the graph.
    """
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
                params[f"omega^{i}"] = se.forward_noise[i]
            if i < len(se.backward_noise) and isinstance(se.backward_noise[i], torch.Tensor):
                params[f"xi^{i}"] = se.backward_noise[i]

            if i < len(se.weight_memory) and isinstance(se.weight_memory[i], torch.Tensor):
                params[f"gamma^[w]_{i}"] = se.weight_memory[i]
            if i < len(se.residual_memory) and isinstance(se.residual_memory[i], torch.Tensor):
                params[f"gamma^[g]_{i}"] = se.residual_memory[i]

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
