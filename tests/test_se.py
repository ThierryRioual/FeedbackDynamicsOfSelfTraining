import torch
from src.config import DataConfig, AlgorithmConfig
from src.objectives import LogisticLoss, RidgePenalty
from src.asymptotics import MacroscopicStateEvolution
from src.dgp import IsotropicGaussian

def main():
    torch.set_default_dtype(torch.float64)
    data_cfg = DataConfig(data_to_dimension_ratio=2.0)
    algo_cfg = AlgorithmConfig(n_iterations=3, loss_function=LogisticLoss(), penalty_function=RidgePenalty(0.1))
    
    se = MacroscopicStateEvolution(data_cfg=data_cfg, algo_cfg=algo_cfg, K=10)
    print("Testing initialization...")
    print(f"Weight history length: {len(se.weight)}")
    print(f"Preactivation history length: {len(se.preactivation)}")
    
    print("Testing step(0)...")
    se.step(0)
    print("step(0) complete")
    
    print("Testing step(1)...")
    se.step(1)
    print("step(1) complete")
    
    try:
        se.step(3)
        print("FAILED: Did not block invalid future step")
    except RuntimeError as e:
        print(f"SUCCESS: Blocked invalid future step: {e}")

if __name__ == "__main__":
    main()
