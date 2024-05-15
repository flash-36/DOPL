import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from dopo.envs import MultiArmRestlessDuellingEnv
from dopo.utils import load_arm
from dopo.train import train, assisted_train
from dopo.train import get_opt_performance
from dopo.plot import plot_training_performance, plot_reconstruction_loss
import logging
import random

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
    env.T = cfg.T

    # Get optimal performance and index matrix
    opt_cost, opt_index = get_opt_performance(env)

    # Perform training and evaluation using parameters
    performances = []
    losses = []
    failure_points = []
    for seeds in range(cfg.num_seeds):
        print("*" * 40, f"Training Seed {seeds+1}", "*" * 40)
        performance, loss, failure_point = train(env, cfg)
        # performance, loss, failure_point = assisted_train(env, cfg)
        performances.append(performance)
        losses.append(loss)
        failure_points.append(failure_point)

    # Plot the training performance
    plot_training_performance(
        performances, opt_cost, min(failure_points), cfg["exp"]["name"]
    )
    plot_reconstruction_loss(losses, cfg["exp"]["name"])


if __name__ == "__main__":
    main()
