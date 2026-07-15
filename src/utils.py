from dgp import SpikedIsotropic

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Type, Optional

def run_experiment(
    data_gen: 'SpikedIsotropic', 
    algorithm: Type,
    N: int, M: int, N_test: int, T: int,
    base_params: Dict[str, Any], 
    seeds: List[int],
    sweep_param_name: Optional[str] = None, 
    sweep_param_values: Optional[List[Any]] = None
) -> np.ndarray:
    """
    Unified master function for executing single self-training runs and sweeps.
    Supports scalars, array-based time schedules, and dynamic margin calculation.
    """
    is_sweep = sweep_param_name is not None and sweep_param_values is not None
    values_to_test = sweep_param_values if is_sweep else [None]
    
    length = len(values_to_test)
    runs = len(seeds)
    res = np.zeros((length, runs, T + 1))

    for j, current_seed in enumerate(seeds):
        
        data_gen.rng = np.random.default_rng(current_seed)
        X_lab, Y_lab, X_unl, X_test, Y_test = data_gen.sample(N, M, N_test)
        
        experiment = algorithm(X_lab, Y_lab, X_unl, X_test, Y_test)

        for i, val in enumerate(values_to_test):
            current_params = base_params.copy()
            
            if is_sweep:
                current_params[sweep_param_name] = val
                
            # 1. Extract 'p' and compute 'kappa' (Supports both scalar and array schedules)
            if 'p' in current_params:
                p = current_params.pop('p')
                p_array = np.asarray(p, dtype=float)
                p_clipped = np.clip(p_array, 1e-9, 1 - 1e-9)
                kappa_logit = np.abs(np.log(p_clipped / (1.0 - p_clipped)))
                current_params['kappa'] = kappa_logit
                
            current_params['T'] = T
            
            # 2. Python keyword collision fix: 'lambda' -> 'lambd'
            if 'lambda' in current_params:
                current_params['lambd'] = current_params.pop('lambda')
            
            # Dynamically unpack all parameters
            errors = experiment.run(**current_params)
            res[i, j, :] = errors

    return res

def plot_experiment(
    res: np.ndarray, 
    base_params: Dict[str, Any], 
    N: int, M: int, N_test: int, T: int,
    algorithm_name: str,
    sweep_param_name: Optional[str] = None, 
    sweep_param_values: Optional[List[Any]] = None
):
    """
    Unified plotting function. Preserves specific LaTeX mapping, conditional parameter
    filtering, and automatically formats time-indexed schedules.
    """
    is_sweep = sweep_param_name is not None and sweep_param_values is not None
    length = len(sweep_param_values) if is_sweep else 1
    
    cmap = plt.colormaps['plasma'] 
    
    latex_map = {
        'gamma': r'$\gamma$',
        'lambd': r'$\lambda$',
        'lambda': r'$\lambda$',
        'p': r'$p$',
        'kappa': r'$\kappa$',
        'eta': r'$\eta$'
    }

    # Helper to prevent large arrays (schedules) from breaking the plot title
    def format_val(val, param_name):
        if isinstance(val, (list, np.ndarray)):
            return "Schedule"
        if isinstance(val, float):
            if param_name == 'p':
                return f"{val:.2f}"
            return f"{val:.3f}"
        return str(val)
    
    fig, axes = plt.subplots(1, length, figsize=(5 * max(1, length), 5), sharey=True)
    if length == 1: axes = [axes]
        
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
            axes[i].set_title("Test Error Trajectory", fontsize=14)
        
        axes[i].plot(np.arange(T+1), mean_trajectory, color=line_color, 
                     linewidth=2.0, label="Mean Error")
        axes[i].plot(np.arange(T+1), median_trajectory, color=line_color, 
                     linewidth=1.0, linestyle='--', alpha=0.8, label="Median Error")
                     
        axes[i].fill_between(np.arange(T+1), lower_bound, upper_bound, 
                             color=line_color, alpha=0.2, label="10th-90th Pct")
        
        axes[i].grid(True, linestyle=':', alpha=0.7)
        axes[i].set_xlabel(r"Iterations ($t$)", fontsize=12)
        
        if i == 0:
            axes[i].legend(loc="upper right", framealpha=0.9)

    axes[0].set_ylabel("Test Error", fontsize=12)
    
    bottom_ylim, top_ylim = axes[0].get_ylim()
    axes[0].set_ylim(max(0.0, bottom_ylim), top_ylim)
    
    # 2. Labeling and Conditional Filtering for the Fixed Parameters
    fixed_str_parts = []
    for k, v in base_params.items():
        if is_sweep and k == sweep_param_name:
            continue
            
        # Filter out 'eta' if the algorithm is not the Gradient Step variant
        if k == 'eta' and 'Gradient' not in algorithm_name:
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