import matplotlib.pyplot as plt
import numpy as np
import hydra
import os
import pandas as pd


def plot_training_performance(performances, opt_cost, failure_point, exp_name):
    train_curves_list = []
    baseline_curves_list = []
    rand_curves_list = []
    opt_curves_list = []
    regret_dopo_list = []
    regret_baseline_list = []
    regret_rand_list = []
    for performance in performances.values():
        # Unpack reward curves
        train_curves_list.append(performance["train_curve"])
        baseline_curves_list.append(performance["baseline_curve"])
        rand_curves_list.append(performance["rand_curve"])
        opt_curves_list.append(performance["opt_curve"])
        # Compute regret based on opt index policy
        regret_dopo_list.append(
            np.array(performance["opt_curve"]) - np.array(performance["train_curve"])
        )
        regret_baseline_list.append(
            np.array(performance["opt_curve"]) - np.array(performance["baseline_curve"])
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
    mean_baseline = np.mean(baseline_curves_list, axis=0)
    std_baseline = np.std(baseline_curves_list, axis=0)

    # smooth the curves with windowed moving avg
    window_size = 10
    mean_curve = np.convolve(mean_curve, np.ones(window_size), "valid") / window_size
    std_curve = np.convolve(std_curve, np.ones(window_size), "valid") / window_size
    mean_rand = np.convolve(mean_rand, np.ones(window_size), "valid") / window_size
    std_rand = np.convolve(std_rand, np.ones(window_size), "valid") / window_size
    mean_opt = np.convolve(mean_opt, np.ones(window_size), "valid") / window_size
    std_opt = np.convolve(std_opt, np.ones(window_size), "valid") / window_size
    mean_baseline = (
        np.convolve(mean_baseline, np.ones(window_size), "valid") / window_size
    )
    std_baseline = (
        np.convolve(std_baseline, np.ones(window_size), "valid") / window_size
    )

    plt.plot(mean_curve, label="DOPL")
    plt.fill_between(
        range(len(mean_curve)),
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.3,
    )
    plt.plot(mean_baseline, color="orange", label="Rew_Est_Index_Baseline")
    plt.fill_between(
        range(len(mean_baseline)),
        mean_baseline - std_baseline,
        mean_baseline + std_baseline,
        color="orange",
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

    # Compute regret and cumulative regret
    mean_rand_regret = np.mean(regret_rand_list, axis=0)
    std_rand_regret = np.std(regret_rand_list, axis=0)
    mean_regret_dopo = np.mean(regret_dopo_list, axis=0)
    std_regret_dopo = np.std(regret_dopo_list, axis=0)
    mean_regret_baseline = np.mean(regret_baseline_list, axis=0)
    std_regret_baseline = np.std(regret_baseline_list, axis=0)

    cum_regret_dopo = np.cumsum(regret_dopo_list, axis=1)
    cum_regret_rand = np.cumsum(regret_rand_list, axis=1)
    cum_regret_baseline = np.cumsum(regret_baseline_list, axis=1)
    mean_cum_regret_dopo = np.mean(cum_regret_dopo, axis=0)
    std_cum_regret_dopo = np.std(cum_regret_dopo, axis=0)
    mean_cum_regret_baseline = np.mean(cum_regret_baseline, axis=0)
    std_cum_regret_baseline = np.std(cum_regret_baseline, axis=0)
    mean_cum_regret_rand = np.mean(cum_regret_rand, axis=0)
    std_cum_regret_rand = np.std(cum_regret_rand, axis=0)

    # smooth the curves with windowed moving avg
    window_size = 10
    mean_regret_dopo = (
        np.convolve(mean_regret_dopo, np.ones(window_size), "valid") / window_size
    )
    std_regret_dopo = (
        np.convolve(std_regret_dopo, np.ones(window_size), "valid") / window_size
    )
    mean_rand_regret = (
        np.convolve(mean_rand_regret, np.ones(window_size), "valid") / window_size
    )
    std_rand_regret = (
        np.convolve(std_rand_regret, np.ones(window_size), "valid") / window_size
    )
    mean_regret_baseline = (
        np.convolve(mean_regret_baseline, np.ones(window_size), "valid") / window_size
    )
    std_regret_baseline = (
        np.convolve(std_regret_baseline, np.ones(window_size), "valid") / window_size
    )
    mean_cum_regret_dopo = (
        np.convolve(mean_cum_regret_dopo, np.ones(window_size), "valid") / window_size
    )
    std_cum_regret_dopo = (
        np.convolve(std_cum_regret_dopo, np.ones(window_size), "valid") / window_size
    )
    mean_cum_regret_rand = (
        np.convolve(mean_cum_regret_rand, np.ones(window_size), "valid") / window_size
    )
    std_cum_regret_rand = (
        np.convolve(std_cum_regret_rand, np.ones(window_size), "valid") / window_size
    )
    mean_cum_regret_baseline = (
        np.convolve(mean_cum_regret_baseline, np.ones(window_size), "valid")
        / window_size
    )

    # Plot regret
    plt.plot(mean_regret_dopo, label="DOPO Regret")
    plt.fill_between(
        range(len(mean_regret_dopo)),
        mean_regret_dopo - std_regret_dopo,
        mean_regret_dopo + std_regret_dopo,
        alpha=0.3,
    )
    plt.plot(
        mean_regret_baseline, color="orange", label="Rew_Est_Index_Baseline Regret"
    )
    plt.fill_between(
        range(len(mean_regret_baseline)),
        mean_regret_baseline - std_regret_baseline,
        mean_regret_baseline + std_regret_baseline,
        color="orange",
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

    # Plot cumulative regret
    plt.plot(mean_cum_regret_dopo, label="DOPO Cumulative Regret")
    plt.fill_between(
        range(len(mean_cum_regret_dopo)),
        mean_cum_regret_dopo - std_cum_regret_dopo,
        mean_cum_regret_dopo + std_cum_regret_dopo,
        alpha=0.3,
    )
    plt.plot(
        mean_cum_regret_baseline,
        color="orange",
        label="Rew_Est_Index_Baseline Cumulative Regret",
    )
    plt.fill_between(
        range(len(mean_cum_regret_baseline)),
        mean_cum_regret_baseline - std_cum_regret_baseline,
        mean_cum_regret_baseline + std_cum_regret_baseline,
        color="orange",
        alpha=0.3,
    )
    plt.plot(mean_cum_regret_rand, color="r", label="Random Baseline Cumulative Regret")
    plt.fill_between(
        range(len(mean_cum_regret_rand)),
        mean_cum_regret_rand - std_cum_regret_rand,
        mean_cum_regret_rand + std_cum_regret_rand,
        color="red",
        alpha=0.3,
    )
    plt.plot
    plt.legend()
    plt.title(f"Cumulative Regret - {exp_name}")
    plt.xlabel("Iterations")
    plt.ylabel("Cumulative Regret")

    # Save the cumulative regret plot to Hydra's output directory
    plt.savefig(os.path.join(output_dir, "cumulative_regret.png"))
    plt.clf()

    # Save performances
    for key, performance in performances.items():
        performance["opt_cost"] = [opt_cost] * len(performance["opt_curve"])
        info_df = pd.DataFrame.from_dict(
            performance,
        )
        info_df.to_csv(os.path.join(output_dir, f"performance_info_seed_{key}.csv"))

    # Save optimal_cost and failure_point
    with open(os.path.join(output_dir, "opt_cost.txt"), "w") as f:
        f.write(str(opt_cost))
    with open(os.path.join(output_dir, "failure_point.txt"), "w") as f:
        f.write(str(failure_point))


def plot_reconstruction_loss(losses, exp_name):
    index_error_dopls = []
    index_error_baselines = []
    F_errors = []
    P_errors = []
    P_errors_baseline = []
    Q_errors = []
    for loss in losses.values():
        index_error_dopls.append(loss["index_error_dopl"])
        index_error_baselines.append(loss["index_error_baseline"])
        F_errors.append(loss["F_error"])
        P_errors.append(loss["P_error_dopl"])
        P_errors_baseline.append(loss["P_error_baseline"])
        Q_errors.append(loss["Q_error"])

    mean_index_error_dopl = np.mean(index_error_dopls, axis=0)
    std_index_error_dopl = np.std(index_error_dopls, axis=0)
    mean_index_error_baseline = np.mean(index_error_baselines, axis=0)
    std_index_error_baseline = np.std(index_error_baselines, axis=0)
    mean_F_error = np.mean(F_errors, axis=0)
    std_F_error = np.std(F_errors, axis=0)
    mean_P_error = np.mean(P_errors, axis=0)
    std_P_error = np.std(P_errors, axis=0)
    mean_P_error_baseline = np.mean(P_errors_baseline, axis=0)
    std_P_error_baseline = np.std(P_errors_baseline, axis=0)
    mean_Q_error = np.mean(Q_errors, axis=0)
    std_Q_error = np.std(Q_errors, axis=0)

    # Smooth the curves with windowed moving avg
    window_size = 10
    mean_index_error_dopl = (
        np.convolve(mean_index_error_dopl, np.ones(window_size), "valid") / window_size
    )
    std_index_error_dopl = (
        np.convolve(std_index_error_dopl, np.ones(window_size), "valid") / window_size
    )
    mean_index_error_baseline = (
        np.convolve(mean_index_error_baseline, np.ones(window_size), "valid")
        / window_size
    )
    std_index_error_baseline = (
        np.convolve(std_index_error_baseline, np.ones(window_size), "valid")
        / window_size
    )
    mean_F_error = (
        np.convolve(mean_F_error, np.ones(window_size), "valid") / window_size
    )
    std_F_error = np.convolve(std_F_error, np.ones(window_size), "valid") / window_size
    mean_P_error = (
        np.convolve(mean_P_error, np.ones(window_size), "valid") / window_size
    )
    std_P_error = np.convolve(std_P_error, np.ones(window_size), "valid") / window_size
    mean_P_error_baseline = (
        np.convolve(mean_P_error_baseline, np.ones(window_size), "valid") / window_size
    )
    std_P_error_baseline = (
        np.convolve(std_P_error_baseline, np.ones(window_size), "valid") / window_size
    )
    mean_Q_error = (
        np.convolve(mean_Q_error, np.ones(window_size), "valid") / window_size
    )
    std_Q_error = np.convolve(std_Q_error, np.ones(window_size), "valid") / window_size

    plt.plot(mean_index_error_dopl, color="b", linestyle="--", label="Index Error")
    plt.fill_between(
        range(len(mean_index_error_dopl)),
        mean_index_error_dopl - std_index_error_dopl,
        mean_index_error_dopl + std_index_error_dopl,
        alpha=0.3,
    )
    plt.plot(
        mean_index_error_baseline,
        color="purple",
        linestyle="--",
        label="Baseline Index Error",
    )
    plt.fill_between(
        range(len(mean_index_error_baseline)),
        mean_index_error_baseline - std_index_error_baseline,
        mean_index_error_baseline + std_index_error_baseline,
        alpha=0.3,
    )
    plt.plot(mean_F_error, color="b", label="F Error")
    plt.fill_between(
        range(len(mean_F_error)),
        mean_F_error - std_F_error,
        mean_F_error + std_F_error,
        color="blue",
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
    plt.plot(mean_P_error_baseline, color="pink", label="Baseline P Error")
    plt.fill_between(
        range(len(mean_P_error_baseline)),
        mean_P_error_baseline - std_P_error_baseline,
        mean_P_error_baseline + std_P_error_baseline,
        color="pink",
        alpha=0.3,
    )
    plt.plot(mean_Q_error, color="orange", label="Q Error")
    plt.fill_between(
        range(len(mean_Q_error)),
        mean_Q_error - std_Q_error,
        mean_Q_error + std_Q_error,
        color="orange",
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

    # Save losses
    for key, loss in losses.items():
        info_df = pd.DataFrame.from_dict(
            loss,
        )
        info_df.to_csv(os.path.join(output_dir, f"loss_info_seed_{key}.csv"))


def plot_meta_data(meta, exp_name):
    delta_tracker_P = []
    delta_tracker_F = []
    for meta_info in meta.values():
        delta_tracker_P.append(meta_info["delta_tracker_P"])
        delta_tracker_F.append(meta_info["delta_tracker_F"])

    mean_delta_tracker_P = np.mean(delta_tracker_P, axis=0)
    std_delta_tracker_P = np.std(delta_tracker_P, axis=0)
    mean_delta_tracker_F = np.mean(delta_tracker_F, axis=0)
    std_delta_tracker_F = np.std(delta_tracker_F, axis=0)

    # Plot the mean and std of meta data
    plt.plot(mean_delta_tracker_P, color="g", label="delta_P")
    plt.fill_between(
        range(len(mean_delta_tracker_P)),
        mean_delta_tracker_P - std_delta_tracker_P,
        mean_delta_tracker_P + std_delta_tracker_P,
        alpha=0.3,
    )
    plt.plot(mean_delta_tracker_F, color="r", label="delta_F")
    plt.fill_between(
        range(len(mean_delta_tracker_F)),
        mean_delta_tracker_F - std_delta_tracker_F,
        mean_delta_tracker_F + std_delta_tracker_F,
        color="red",
        alpha=0.3,
    )
    # Plot the minimum delta at each iteration for each seed
    min_delta_tracker_P = np.min(delta_tracker_P, axis=0)
    min_delta_tracker_F = np.min(delta_tracker_F, axis=0)
    plt.plot(min_delta_tracker_P, color="g", linestyle="--", label="min_delta_P")
    plt.plot(min_delta_tracker_F, color="r", linestyle="--", label="min_delta_F")

    plt.legend()
    plt.title(f"Meta Data - {exp_name}")
    plt.xlabel("Iterations")
    plt.ylabel("delta")

    # Save the plots to Hydra's output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    plt.savefig(os.path.join(output_dir, "meta_data.png"))
    plt.clf()
    elp_cost_tracker = []
    for meta_info in meta.values():
        elp_cost_tracker.append(meta_info["elp_cost_tracker"])
    mean_elp_cost = np.mean(elp_cost_tracker, axis=0)
    std_elp_cost = np.std(elp_cost_tracker, axis=0)
    plt.plot(mean_elp_cost, label="ELP Cost")
    plt.fill_between(
        range(len(mean_elp_cost)),
        mean_elp_cost - std_elp_cost,
        mean_elp_cost + std_elp_cost,
        alpha=0.3,
    )
    plt.title(f"ELP Cost - {exp_name}")
    plt.xlabel("Iterations")
    plt.ylabel("ELP Opt Cost")

    # Save the plots to Hydra's output directory
    plt.savefig(os.path.join(output_dir, "elp_cost.png"))
    plt.clf()

    # Save meta data
    for key, meta_info in meta.items():
        info_df = pd.DataFrame.from_dict(
            meta_info,
        )
        info_df.to_csv(os.path.join(output_dir, f"meta_info_seed_{key}.csv"))
