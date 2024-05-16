import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from dopo.envs import MultiArmRestlessDuellingEnv
from dopo.utils import load_arm
from dopo.train import train, assisted_train
from dopo.train import get_opt_performance
from dopo.plot import (
    plot_training_performance,
    plot_reconstruction_loss,
    plot_meta_data,
)
import logging
import random
from collections import defaultdict
import warnings
import wandb
from dopo.utils import set_seed

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="toy", version_base=None)
def main(cfg: DictConfig):
    # Extract environment configuration values
    env_config = cfg.env_config
    arm_prefix = env_config.arm
    num_types = env_config.num_types
    num_arms_per_type = env_config.num_arms_per_type
    arm_constraint = env_config.arm_constraint

    # Prepare lists for transition (P) and reward (R) matrices
    P_list = []
    R_list = []
    for i in range(1, num_types + 1):
        P, R = load_arm(f"{arm_prefix}_arm_type_{i}")
        P_list.extend([P] * num_arms_per_type)
        R_list.extend([R] * num_arms_per_type)
    # Shffle ordering of arms so as to not have biases
    indices = list(range(len(P_list)))
    random.seed(cfg.num_seeds)
    random.shuffle(indices)
    P_list = [P_list[i] for i in indices]
    R_list = [R_list[i] for i in indices]
    # Check that the arm constraint does not exceed the number of arms
    total_arms = num_types * num_arms_per_type
    assert (
        arm_constraint < total_arms
    ), "arm_constraint exceeds the number of arms available."

    # Initialize the duelling environment
    env = MultiArmRestlessDuellingEnv(arm_constraint, P_list, R_list)
    env.H = cfg.H

    # Get optimal performance and index matrix
    opt_cost = get_opt_performance(env)

    # Perform training and evaluation using parameters
    performances = {}
    losses = {}
    failure_points = []
    metas = {}
    for seeds in range(cfg.num_seeds):
        set_seed(seeds)
        print("*" * 40, f"Training Seed {seeds+1}", "*" * 40)
        wandb.init(project="dopo", name=f"{cfg.exp.name}_seed_{seeds+1}")
        wandb.log({"env_opt_reward": opt_cost})
        performance, loss, meta, failure_point = train(env, cfg, seeds)
        performances[seeds] = performance
        losses[seeds] = loss
        metas[seeds] = meta
        failure_points.append(failure_point)
        wandb.finish()

    # Plot the training performance
    plot_training_performance(
        performances, opt_cost, min(failure_points), cfg["exp"]["name"]
    )
    # Plot the meta data
    plot_meta_data(metas, cfg["exp"]["name"])
    # Plot the reconstruction loss
    plot_reconstruction_loss(losses, cfg["exp"]["name"])


if __name__ == "__main__":
    main()
