from dopo.envs import MultiArmRestlessDuellingEnv
import random
import wandb
import os
import hydra
import pandas as pd
import numpy as np
from pathlib import Path
import json

RMAB_PATH = os.path.join(Path(__file__).parent.parent, "RMAB_env_instances")


def set_seed(seed):
    np.random.seed(seed)


def load_arm(arm_name):
    P = np.load(os.path.join(RMAB_PATH, f"{arm_name}_transitions.npy"))
    R = np.load(os.path.join(RMAB_PATH, f"{arm_name}_rewards.npy"))
    return P, R


def load_environment_configuration(cfg):
    """Load transition kernels and reward matrix for each arm and shuffle arms (to prevent biases)."""
    env_config = cfg.env_config
    arm_prefix = env_config.arm
    num_types = env_config.num_types
    num_arms_per_type = env_config.num_arms_per_type
    arm_constraint = env_config.arm_constraint

    P_list = []
    R_list = []
    for i in range(1, num_types + 1):
        P, R = load_arm(f"{arm_prefix}_arm_type_{i}")
        P_list.extend([P] * num_arms_per_type)
        R_list.extend([R] * num_arms_per_type)

    indices = list(range(len(P_list)))
    random.seed(0)
    random.shuffle(indices)
    P_list = [P_list[i] for i in indices]
    R_list = [R_list[i] for i in indices]

    total_arms = num_types * num_arms_per_type
    assert arm_constraint < total_arms, "arm_constraint not meaningful."

    return P_list, R_list, arm_constraint


def initialize_environment(cfg, P_list, R_list, arm_constraint):
    """Initialize the Pref-RMAB environment."""
    env = MultiArmRestlessDuellingEnv(arm_constraint, P_list, R_list)
    env.H = cfg.H
    return env


# def save_results(results_dict, seeds):
#     """Save results to disk"""
#     results_dict = pd.DataFrame(results_dict)
#     hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
#     output_dir = hydra_cfg.runtime.output_dir
#     results_dict.to_csv(os.path.join(output_dir, f"results_{seeds}.csv"))


# def load_results():
#     """Load results from disk for all seeds"""
#     hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
#     output_dir = hydra_cfg.runtime.output_dir
#     all_results = []

#     for filename in os.listdir(output_dir):
#         if filename.startswith("results_") and filename.endswith(".csv"):
#             file_path = os.path.join(output_dir, filename)
#             df = pd.read_csv(file_path)
#             all_results.append(df)

#     if all_results:
#         return all_results
#     else:
#         raise FileNotFoundError("No results found in the output directory.")


def save_results(results_dict, seeds):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    file_path = os.path.join(output_dir, f"results_{seeds}.json")
    with open(file_path, "w") as json_file:
        json.dump(results_dict, json_file, indent=4)


def load_results():
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    all_results = []
    for filename in os.listdir(output_dir):
        if filename.startswith("results_") and filename.endswith(".json"):
            file_path = os.path.join(output_dir, filename)
            with open(file_path, "r") as json_file:
                results_dict = json.load(json_file)
                all_results.append(results_dict)
    if all_results:
        for i, result in enumerate(all_results):
            print(f"Loaded results for seed {i}: {result}")
        return all_results
    else:
        raise FileNotFoundError("No results found in the output directory.")


def wandb_log_latest(metrics, step):
    """Log metrics to wandb."""
    for key, value in metrics.items():
        wandb.log({key: value[-1]}, step=step)
