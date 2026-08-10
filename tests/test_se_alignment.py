import torch
import math

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

from src.config import DataConfig, AlgorithmConfig
from src.dgp import IsotropicGaussian
from src.algorithms import SelfTrainedGradientDescent
from src.asymptotics import MacroscopicStateEvolution

def run_alignment_test():
    print("=== SE vs GD Alignment Test ===")
    
    # 1. Environment & Initialization Setup
    N = 3000
    d = 1000
    K = 50000
    
    data_cfg = DataConfig(
        scale=1.0,
        label_prior=0.5,
        observation_prior=0.5,
        data_to_dimension_ratio=N/d,
        signal_prior=lambda: torch.randn(1).item()
    )
    
    algo_cfg = AlgorithmConfig(
        n_iterations=5,
        margin_threshold=0.0,
        step_size=0.1,
        penalty_param=0.1,
        pseudo_label_param=1.0,
        ramp_start=0,
        ramp_end=1,
        include_bias=False
    )
    
    # Init DGP and GD dataset
    dgp = IsotropicGaussian(cfg=data_cfg, n_train=N, n_test=1000, dimensions=d, seed=42)
    X_lab, Y_lab, X_unl, Y_unl, X_test, Y_test = dgp.sample(stratified=False)
    mu_gd = dgp._mu 
    
    # Init GD
    gd = SelfTrainedGradientDescent(cfg=algo_cfg)
    w_0 = torch.randn(d, dtype=torch.float64)
    gd.bias = 0.0
    gd.weights = w_0.clone()
    
    # Init SE
    se = MacroscopicStateEvolution(
        data_cfg=data_cfg,
        algo_cfg=algo_cfg,
        mc_seed=42,
        K=K,
        initial_bias=0.0,
        initial_weight=None
    )
    se.__post_init__() # Make sure to run post init if not running dataclass init normally
    
    # --- CHECKPOINT 1: t=0 (Initialization) ---
    print("\n[Checkpoint 1: t=0 (Initialization)]")
    
    gd_w0_norm = (torch.linalg.norm(gd.weights)**2 / d).item()
    
    se.weight[0] = se.initial_weight
    se.bias[0] = se.initial_bias
    se_w0_norm = (torch.linalg.norm(se.weight[0])**2 / K).item()
    
    print(f"Weight Norm (||w||^2) | GD: {gd_w0_norm:.6f} | SE: {se_w0_norm:.6f} | Diff: {abs(gd_w0_norm - se_w0_norm):.6f}")
    
    # Depending on how signal is scaled in DGP vs SE, we compute overlap.
    # User requested: (mu_gd @ w^0) / d for GD (Assuming x_target is mu)
    m0_gd = (torch.dot(mu_gd * math.sqrt(d), gd.weights) / d).item() # Scaling mu to match SE variance 1
    m0_se = se.compute_weight_signal_alignment(0)
    print(f"Overlap (m_0)         | GD: {m0_gd:.6f} | SE: {m0_se:.6f} | Diff: {abs(m0_gd - m0_se):.6f}")


    # --- CHECKPOINT 2: t=1 (First Forward Pass) ---
    print("\n[Checkpoint 2: t=1 (First Forward Pass)]")
    
    # GD Forward Pass
    X_total = torch.cat([X_lab, X_unl])
    r_gd = (X_total @ gd.weights) / math.sqrt(d) + gd.bias
    r_gd_mean = r_gd.mean().item()
    r_gd_var = r_gd.var(unbiased=False).item()
    
    # SE Forward Pass
    se.forward_pass(0)
    r_se = se.preactivation[0]
    r_se_mean = r_se.mean().item()
    r_se_var = r_se.var(unbiased=False).item()
    
    print(f"Preactivation Mean    | GD: {r_gd_mean:.6f} | SE: {r_se_mean:.6f} | Diff: {abs(r_gd_mean - r_se_mean):.6f}")
    print(f"Preactivation Var     | GD: {r_gd_var:.6f} | SE: {r_se_var:.6f} | Diff: {abs(r_gd_var - r_se_var):.6f}")


    # --- CHECKPOINT 3: t=1 (First Backward Pass & Weight Update) ---
    print("\n[Checkpoint 3: t=1 (First Backward Pass & Weight Update)]")
    
    # GD Backward Pass (1 step of GD)
    alpha = algo_cfg.get_pseudo_label_weight(0)
    emp_risk, weight_grad = gd._compute_gradient(gd.bias, gd.weights, X_lab, Y_lab, X_unl, alpha)
    # Update GD weights (assuming standard gradient descent step W_{t+1} = W_t - eta * grad)
    gd.weights = gd.weights - algo_cfg.step_size * weight_grad
    
    # SE Backward Pass
    se.backward_pass(0)
    se._current_t += 1
    
    gd_w1_norm = (torch.linalg.norm(gd.weights)**2 / d).item()
    se_w1_norm = (torch.linalg.norm(se.weight[1])**2 / K).item()
    
    m1_gd = (torch.dot(mu_gd * math.sqrt(d), gd.weights) / d).item()
    m1_se = se.compute_weight_signal_alignment(1)
    
    print(f"Weight Norm (||w^1||^2)| GD: {gd_w1_norm:.6f} | SE: {se_w1_norm:.6f} | Diff: {abs(gd_w1_norm - se_w1_norm):.6f}")
    print(f"Overlap (m_1)         | GD: {m1_gd:.6f} | SE: {m1_se:.6f} | Diff: {abs(m1_gd - m1_se):.6f}")


    # --- CHECKPOINT 4: t=2 (Second Forward Pass - The Onsager Check) ---
    print("\n[Checkpoint 4: t=2 (Second Forward Pass - The Onsager Check)]")
    
    # GD Second Forward Pass
    r_gd_2 = (X_total @ gd.weights) / math.sqrt(d) + gd.bias
    r_gd_var_2 = r_gd_2.var(unbiased=False).item()
    
    # SE Second Forward Pass
    se.forward_pass(1)
    r_se_2 = se.preactivation[1]
    r_se_var_2 = r_se_2.var(unbiased=False).item()
    
    print(f"Preactivation Var (t=2) | GD: {r_gd_var_2:.6f} | SE: {r_se_var_2:.6f} | Diff: {abs(r_gd_var_2 - r_se_var_2):.6f}")

if __name__ == "__main__":
    run_alignment_test()
