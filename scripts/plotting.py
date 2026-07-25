import torch
import numpy as np
import matplotlib.pyplot as plt
import textwrap

from typing import Dict, List, Any, Optional

def plot_experiment(
    res: torch.Tensor, 
    base_params: Dict[str, Any], 
    d: int, N: int, M: int, N_test: int, T: int,
    algorithm_name: str,
    sweep_param_name: Optional[str] = None, 
    sweep_param_values: Optional[List[Any]] = None,
    metric_name: str = "test_error",
    extra_text: Optional[str] = None  # NEW: Pass custom notes here!
):
    """
    Unified plotting function. Converts incoming PyTorch tensors to NumPy for Matplotlib.
    Preserves specific LaTeX mapping, conditional parameter filtering, 
    and automatically formats time-indexed schedules.
    """
    if isinstance(res, torch.Tensor):
        res = res.cpu().numpy()

    is_sweep = sweep_param_name is not None and sweep_param_values is not None
    length = len(sweep_param_values) if is_sweep else 1
    
    cmap = plt.colormaps['plasma'] 
    
    latex_map = {
        'gamma': r'$\gamma$',
        'penalty_param': r'$\lambda$',
        'lambd': r'$\lambda$',
        'margin_threshold': r'$\kappa$',
        'kappa': r'$\kappa$',
        'step_size': r'$\eta$',
        'eta': r'$\eta$',
        'pseudo_label_param': r'$\alpha$',
        'alpha': r'$\alpha$',
        'ramp_start': r'$T_0$',
        'ramp_end': r'$T_1$',
        'n_iterations': r'$T$',
        'include_bias': 'Include\xa0bias',
        'loss_function': 'Loss\xa0Function',
        'penalty_function': 'Penalty\xa0Function'
    }

    # FIXED: Catches custom objects to prevent memory address printing
    def format_val(val, param_name):
        if isinstance(val, (list, np.ndarray, torch.Tensor)):
            return "Schedule"
        if isinstance(val, float):
            return f"{val:.3f}"
        # If it's a custom class (like LogisticLoss), get its clean name
        if hasattr(val, '__class__') and type(val).__module__ != 'builtins':
            return val.__class__.__name__
        return str(val)
    
    fig, axes = plt.subplots(1, length, figsize=(5 * max(1, length), 6), sharey=True)
    if length == 1: 
        axes = [axes]
        
    for i in range(length):
        results = res[i]
        
        mean_trajectory = np.mean(results, axis=0)
        median_trajectory = np.median(results, axis=0)
        lower_bound = np.percentile(results, 10, axis=0)
        upper_bound = np.percentile(results, 90, axis=0)

        color_index = i / max(1, length)
        line_color = cmap(color_index)

        if is_sweep:
            val = sweep_param_values[i]
            sym = latex_map.get(sweep_param_name, sweep_param_name)
            val_str = format_val(val, sweep_param_name)
            axes[i].set_title(rf"{sym} = {val_str}", fontsize=14)
        else:
            formatted_title = metric_name.replace('_', ' ').title()
            axes[i].set_title(f"{formatted_title} Trajectory", fontsize=14)
        
        axes[i].plot(np.arange(T+1), mean_trajectory, color=line_color, 
                     linewidth=1.0, label="Mean")
        axes[i].plot(np.arange(T+1), median_trajectory, color=line_color, 
                     linewidth=0.8, linestyle='--', alpha=0.8, label="Median")
                     
        axes[i].fill_between(np.arange(T+1), lower_bound, upper_bound, 
                             color=line_color, alpha=0.15, label="10th-90th Pct")
        
        axes[i].grid(which='major', color='#999999', linestyle='-', linewidth=0.8)
        axes[i].grid(which='minor', color='#999999', linestyle=':', linewidth=0.5)
        axes[i].minorticks_on()

        axes[i].set_xlabel(r"Iterations ($t$)", fontsize=12)
        
        if i == 0:
            axes[i].legend(loc="upper right", framealpha=0.9)

    formatted_ylabel = metric_name.replace('_', ' ').title()
    axes[0].set_ylabel(formatted_ylabel, fontsize=12)
    
    bottom_ylim, top_ylim = axes[0].get_ylim()
    axes[0].set_ylim(max(0.0, bottom_ylim), top_ylim)
    
    # Generate the string of fixed parameters
    fixed_str_parts = []
    for k, v in base_params.items():
        if is_sweep and k == sweep_param_name:
            continue
        if k in ('eta', 'step_size') and 'Gradient' not in algorithm_name:
            continue

        sym_k = latex_map.get(k, k)
        val_str = format_val(v, k)
        fixed_str_parts.append(rf"{sym_k}={val_str}")
    
    fixed_str = ", ".join(fixed_str_parts)
    
    # 1. Clean up the main title (Moved fixed params out!)
    if is_sweep:
        super_sym = latex_map.get(sweep_param_name, sweep_param_name)
        title_prefix = rf"{algorithm_name} - Varying: {super_sym}"
    else:
        title_prefix = rf"{algorithm_name} Dynamics"
        
    fig.suptitle(
        f"{title_prefix} \n with $d$={d}, $N={N}$, $M={M}$, $N_{{test}}={N_test}$ fixed.", 
        fontsize=20, y=1.05
    )
    
    # 2. Build the Text Box contents
    # Wrap the fixed parameters so they don't run off the screen
    wrapped_fixed = textwrap.fill(f"Fixed: {fixed_str}", width=100)
    
    box_text = wrapped_fixed
    if extra_text:
        # Wrap the extra text as well, separated by a newline
        wrapped_extra = textwrap.fill(f"Note: {extra_text}", width=100)
        box_text += f"\n{wrapped_extra}"

    # 3. Adjust plot spacing to make room at the bottom
    #plt.tight_layout()
    fig.subplots_adjust(bottom=0.25) # Shrink the plots up slightly
    
    # 4. Render the Text Box
    fig.text(
        0.5, 0.05, box_text, 
        ha='center', va='bottom', fontsize=16,
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#dee2e6', alpha=0.9)
    )
    
    plt.show()