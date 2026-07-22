from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

def plot_experiment(
    res: np.ndarray, 
    base_params: Dict[str, Any], 
    N: int, M: int, N_test: int, T: int,
    algorithm_name: str,
    sweep_param_name: Optional[str] = None, 
    sweep_param_values: Optional[List[Any]] = None,
    metric_name: str = "test_error"
):
    """
    Unified plotting function. Preserves specific LaTeX mapping, conditional parameter
    filtering, and automatically formats time-indexed schedules.
    """
    is_sweep = sweep_param_name is not None and sweep_param_values is not None
    length = len(sweep_param_values) if is_sweep else 1
    
    cmap = plt.colormaps['plasma'] 
    
    # Updated mapping to handle strict dataclass variable names
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
        'n_iterations': r'$T$'
    }

    # Helper to prevent large arrays (schedules) from breaking the plot title
    def format_val(val, param_name):
        if isinstance(val, (list, np.ndarray)):
            return "Schedule"
        if isinstance(val, float):
            return f"{val:.3f}"
        return str(val)
    
    fig, axes = plt.subplots(1, length, figsize=(5 * max(1, length), 5), sharey=True)
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

        # 1. Title Formatting
        if is_sweep:
            val = sweep_param_values[i]
            sym = latex_map.get(sweep_param_name, sweep_param_name)
            val_str = format_val(val, sweep_param_name)
            axes[i].set_title(rf"{sym} = {val_str}", fontsize=14)
        else:
            formatted_title = metric_name.replace('_', ' ').title()
            axes[i].set_title(f"{formatted_title} Trajectory", fontsize=14)
        
        axes[i].plot(np.arange(T+1), mean_trajectory, color=line_color, 
                     linewidth=2.0, label="Mean")
        axes[i].plot(np.arange(T+1), median_trajectory, color=line_color, 
                     linewidth=1.0, linestyle='--', alpha=0.8, label="Median")
                     
        axes[i].fill_between(np.arange(T+1), lower_bound, upper_bound, 
                             color=line_color, alpha=0.2, label="10th-90th Pct")
        
        axes[i].grid(True, linestyle=':', alpha=0.7)
        axes[i].set_xlabel(r"Iterations ($t$)", fontsize=12)
        
        if i == 0:
            axes[i].legend(loc="upper right", framealpha=0.9)

    # Dynamically label the Y-axis based on the metric requested
    formatted_ylabel = metric_name.replace('_', ' ').title()
    axes[0].set_ylabel(formatted_ylabel, fontsize=12)
    
    bottom_ylim, top_ylim = axes[0].get_ylim()
    axes[0].set_ylim(max(0.0, bottom_ylim), top_ylim)
    
    # 2. Labeling and Conditional Filtering for the Fixed Parameters
    fixed_str_parts = []
    for k, v in base_params.items():
        if is_sweep and k == sweep_param_name:
            continue
            
        # Filter out step_size/eta if the algorithm is not a gradient method
        if k in ('eta', 'step_size') and 'Gradient' not in algorithm_name:
            continue

        sym_k = latex_map.get(k, k)
        val_str = format_val(v, k)
        fixed_str_parts.append(rf"{sym_k}={val_str}")
    
    fixed_str = ", ".join(fixed_str_parts)
    
    # 3. Dynamic Super-Title Generation
    if is_sweep:
        super_sym = latex_map.get(sweep_param_name, sweep_param_name)
        title_prefix = rf"{algorithm_name} varying {super_sym}"
    else:
        title_prefix = rf"{algorithm_name} Dynamics"
        
    fig.suptitle(
        rf"{title_prefix} | Fixed: {fixed_str} | $N={N}$, $M={M}$  $N_{{test}} = {N_test}$", 
        fontsize=16, y=1.05
    )
    
    plt.tight_layout()
    plt.show()