import matplotlib.pyplot as plt
import numpy as np
import hydra
import os


def plot_training_performance(performances, opt_cost):
    train_curves_list = []
    rand_curves_list = []
    for performance in performances:
        train_curves_list.append(performance["train_curve"])
        rand_curves_list.append(performance["rand_curve"])

    mean_rand = np.mean(rand_curves_list, axis=0)
    std_rand = np.std(rand_curves_list, axis=0)
    mean_curve = np.mean(train_curves_list, axis=0)
    std_curve = np.std(train_curves_list, axis=0)

    plt.plot(mean_curve, label="DOPO")
    plt.fill_between(
        range(len(mean_curve)),
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.3,
    )
    plt.plot(mean_rand, color="r", label="Random Baseline")
    plt.fill_between(
        range(len(mean_rand)),
        mean_rand - std_rand,
        mean_rand + std_rand,
        color="red",
        alpha=0.3,
    )
    plt.axhline(opt_cost, color="g", linestyle="--", label="Optimal Cost")

    plt.legend()
    plt.title("Training Performance")
    plt.xlabel("Iterations")
    plt.ylabel("Performance")

    # Save the plots to Hydra's output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    plt.savefig(os.path.join(output_dir, "performance.png"))
    plt.clf()


def plot_reconstruction_loss(losses):
    index_errors = []
    F_errors = []
    P_errors = []
    for loss in losses:
        index_errors.append(loss["index_error"])
        F_errors.append(loss["F_error"])
        P_errors.append(loss["P_error"])

    mean_index_error = np.mean(index_errors, axis=0)
    std_index_error = np.std(index_errors, axis=0)
    mean_F_error = np.mean(F_errors, axis=0)
    std_F_error = np.std(F_errors, axis=0)
    mean_P_error = np.mean(P_errors, axis=0)
    std_P_error = np.std(P_errors, axis=0)

    plt.plot(mean_index_error, color="b", label="Index Error")
    plt.fill_between(
        range(len(mean_index_error)),
        mean_index_error - std_index_error,
        mean_index_error + std_index_error,
        alpha=0.3,
    )
    plt.plot(mean_F_error, color="r", label="F Error")
    plt.fill_between(
        range(len(mean_F_error)),
        mean_F_error - std_F_error,
        mean_F_error + std_F_error,
        color="red",
        alpha=0.3,
    )
    plt.plot(mean_P_error, color="g", label="P Error")
    plt.fill_between(
        range(len(mean_P_error)),
        mean_P_error - std_P_error,
        mean_P_error + std_P_error,
        color="green",
        alpha=0.3,
    )

    plt.legend()
    plt.title("Reconstruction Losses")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    # Save the plots to Hydra's output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    plt.savefig(os.path.join(output_dir, "losses.png"))
    plt.clf()
