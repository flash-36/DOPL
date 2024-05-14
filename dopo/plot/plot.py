import matplotlib.pyplot as plt
import numpy as np
import hydra
import os


def plot_training_performance(performances, opt_cost, failure_point, exp_name):
    train_curves_list = []
    rand_curves_list = []
    opt_curves_list = []
    regret_dopo_list = []
    regret_rand_list = []
    for performance in performances:
        # Unpack reward curves
        train_curves_list.append(performance["train_curve"])
        rand_curves_list.append(performance["rand_curve"])
        opt_curves_list.append(performance["opt_curve"])
        # Compute regret based on opt index policy
        regret_dopo_list.append(
            np.array(performance["opt_curve"]) - np.array(performance["train_curve"])
        )
        regret_rand_list.append(
            np.array(performance["opt_curve"]) - np.array(performance["rand_curve"])
        )

    mean_rand = np.mean(rand_curves_list, axis=0)
    std_rand = np.std(rand_curves_list, axis=0)
    mean_curve = np.mean(train_curves_list, axis=0)
    std_curve = np.std(train_curves_list, axis=0)
    mean_opt = np.mean(opt_curves_list, axis=0)
    std_opt = np.std(opt_curves_list, axis=0)

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
    plt.plot(mean_opt, color="g", label="Optimal Policy")
    plt.fill_between(
        range(len(mean_opt)),
        mean_opt - std_opt,
        mean_opt + std_opt,
        color="green",
        alpha=0.3,
    )
    plt.axhline(opt_cost, color="g", linestyle="--", label="Optimal Cost")
    plt.axvline(failure_point, color="gray", linestyle="-.", label="Failure Point")

    plt.legend()
    plt.title(f"Training Performance - {exp_name}")
    plt.xlabel("Iterations")
    plt.ylabel("Performance")

    # Save the performance plot to Hydra's output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    plt.savefig(os.path.join(output_dir, "performance.png"))
    plt.clf()

    mean_rand_regret = np.mean(regret_rand_list, axis=0)
    std_rand_regret = np.std(regret_rand_list, axis=0)
    mean_regret_dopo = np.mean(regret_dopo_list, axis=0)
    std_regret_dopo = np.std(regret_dopo_list, axis=0)

    plt.plot(mean_regret_dopo, label="DOPO Regret")
    plt.fill_between(
        range(len(mean_regret_dopo)),
        mean_regret_dopo - std_regret_dopo,
        mean_regret_dopo + std_regret_dopo,
        alpha=0.3,
    )
    plt.plot(mean_rand_regret, color="r", label="Random Baseline Regret")
    plt.fill_between(
        range(len(mean_rand_regret)),
        mean_rand_regret - std_rand_regret,
        mean_rand_regret + std_rand_regret,
        color="red",
        alpha=0.3,
    )
    plt.legend()
    plt.title(f"Regret - {exp_name}")
    plt.xlabel("Iterations")
    plt.ylabel("Regret")

    # Save the regret plot (traj based) to Hydra's output directory
    plt.savefig(os.path.join(output_dir, "regret.png"))
    plt.clf()


def plot_reconstruction_loss(losses, exp_name):
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
    plt.title(f"Reconstruction Losses - {exp_name}")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    # Save the plots to Hydra's output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    plt.savefig(os.path.join(output_dir, "losses.png"))
    plt.clf()
