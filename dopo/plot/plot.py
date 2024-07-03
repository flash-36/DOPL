import matplotlib.pyplot as plt
import numpy as np
import hydra
import os
import pandas as pd
from dopo.utils import load_results


def plot_training_performance(cfg):
    window_size = cfg["plotting"]["window_size"]
    colors = cfg["plotting"]["algo_colors"]
    algos = cfg["algos"]
    color_map = {algo: color for algo, color in zip(algos, colors)}
    results_dicts = load_results()
    for results_dict in results_dicts:
        for algo_name, results in results_dict.items():
            if algo_name != "oracle":
                results["regret"] = np.array(
                    results_dict["oracle"]["reward"]
                ) - np.array(results["reward"])
                results["regret"] = results["regret"].cumsum()

    # Plot reward curves
    fig, axes = plt.subplots(1, 2, figsize=(30, 10))
    for algo_name in results_dicts[0].keys():
        rewards = [
            list(results_dict[algo_name]["reward"]) for results_dict in results_dicts
        ]
        mean_reward = np.mean(rewards, axis=0)
        std_reward = np.std(rewards, axis=0)
        # Apply moving average for smoothing
        mean_curve = (
            pd.Series(mean_reward).rolling(window=window_size, min_periods=1).mean()
        )
        std_curve = (
            pd.Series(std_reward).rolling(window=window_size, min_periods=1).mean()
        )

        # Convert to float explicitly and to numpy arrays
        mean_curve = np.array(mean_curve.astype(float))
        std_curve = np.array(std_curve.astype(float))

        x_values = range(len(mean_curve))
        axes[0].plot(
            x_values,
            mean_curve,
            label=f"{algo_name}",
            color=color_map[algo_name],
        )
        axes[0].fill_between(
            x_values,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
            color=color_map[algo_name],
        )

    axes[0].set_xlabel("Episodes")
    axes[0].set_ylabel("Episodic Reward")
    axes[0].set_title("Reward", fontsize=50)

    # Plot regret curves
    fig, axes = plt.subplots(1, 2, figsize=(30, 10))
    for algo_name in results_dicts[0].keys() - {"oracle"}:
        regrets = [
            list(results_dict[algo_name]["regret"]) for results_dict in results_dicts
        ]
        mean_regret = np.mean(regrets, axis=0)
        std_regret = np.std(regrets, axis=0)
        # Apply moving average for smoothing
        mean_curve = (
            pd.Series(mean_regret).rolling(window=window_size, min_periods=1).mean()
        )
        std_curve = (
            pd.Series(std_regret).rolling(window=window_size, min_periods=1).mean()
        )

        # Convert to float explicitly and to numpy arrays
        mean_curve = np.array(mean_curve.astype(float))
        std_curve = np.array(std_curve.astype(float))

        x_values = range(len(mean_curve))
        axes[1].plot(
            x_values,
            mean_curve,
            label=f"{algo_name}",
            color=color_map[algo_name],
        )
        axes[1].fill_between(
            x_values,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
            color=color_map[algo_name],
        )

    axes[1].set_xlabel("Episodes")
    axes[1].set_ylabel("Cumulative Regret")
    axes[1].set_title("Regret", fontsize=50)

    # Adding a single legend above the plots
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.0),
        fontsize=44,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout to prevent label cutoff
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    plt.savefig(
        os.path.join(output_dir, f"reward_regret_{cfg['env_config']['arm']}.pdf"),
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_reconstruction_loss(cfg):
    pass  # TODO
