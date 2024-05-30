import numpy as np
from tqdm import tqdm
from dopo.utils import compute_ELP, compute_ELP_pyomo, compute_optimal_pyomo
from dopo.train.helpers import apply_index_policy, compute_F_true
from dopo.train.mle import mle_bradley_terry
from dopo.train.mle_grad_desc import mle_gradient_descent
import logging
import wandb
from dopo.utils import set_seed
import time

log = logging.getLogger(__name__)
# delta_scheduler = [0.5] * 2000 + [0.2] * 2000 + [0.1] * 20000 + [0.05] * 20000


def pick_best_ref(W):
    ref_arm = 0
    ref_state = 0
    for arm in range(W.shape[0]):
        for state in range(W.shape[1]):
            if W[arm, state].sum() > W[ref_arm, ref_state].sum():
                ref_arm = arm
                ref_state = state
    return ref_arm, 0


def pick_random_ref(W):
    ref_arm = np.random.randint(0, W.shape[0])
    ref_state = np.random.randint(0, W.shape[1])
    return ref_arm, ref_state


def enrich_F(F_tilde, F_hat, reference, conf):
    ref_arm, ref_state = reference
    num_arms = F_hat.shape[0]
    num_states = F_hat.shape[1]
    # Use lemma 4 from paper; j1 = ref_arm, ref_state; j2 = arm, state
    # Pick best i
    best_arm_i, best_state_i = -1, -1
    best_conf_i = np.inf
    for arm_i in range(num_arms):
        for state_i in range(num_states):
            if conf[ref_arm, ref_state, arm_i, state_i] < best_conf_i:
                best_arm_i, best_state_i = arm_i, state_i
                best_conf_i = conf[ref_arm, ref_state, arm_i, state_i]
    # Use best i to enrich F
    for arm in range(num_arms):
        for state in range(num_states):
            conf_inferred = (
                best_conf_i + conf[arm, state, best_arm_i, best_state_i]
            ) * 1.3
            if conf_inferred < conf[ref_arm, ref_state, arm, state]:

                F_hat[ref_arm, ref_state, arm, state] = (
                    F_hat[ref_arm, ref_state, best_arm_i, best_state_i]
                    * F_hat[best_arm_i, best_state_i, arm, state]
                ) / (
                    F_hat[ref_arm, ref_state, best_arm_i, best_state_i]
                    * F_hat[best_arm_i, best_state_i, arm, state]
                    + F_hat[arm, state, best_arm_i, best_state_i]
                    * F_hat[best_arm_i, best_state_i, ref_arm, ref_state]
                )
                F_hat[arm, state, ref_arm, ref_state] = (
                    1 - F_hat[ref_arm, ref_state, arm, state]
                )
                conf[arm, state, ref_arm, ref_state] = conf[
                    ref_arm, ref_state, arm, state
                ] = conf_inferred
                F_tilde[arm, state, ref_arm, ref_state] = (
                    F_hat[arm, state, ref_arm, ref_state] + conf_inferred
                )
                F_tilde[ref_arm, ref_state, arm, state] = (
                    F_hat[ref_arm, ref_state, arm, state] + conf_inferred
                )
                wandb.log({"successful_enrichment": 1})
            else:
                wandb.log({"successful_enrichment": 0})

    return F_tilde, F_hat


def train(env, cfg, seeds):
    # Extract training parameters
    set_seed(seeds)
    K = cfg["K"]
    delta_coeff = cfg["delta_coeff"]
    conf_coeff = cfg["conf_coeff"]

    # True values
    P_true = np.array(env.P_list)
    F_true = compute_F_true(env)
    R_true = np.array(env.R_list)[:, :, 0]

    # Initialize utility variables
    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    num_actions = env.R_list[0].shape[1]
    # Initialze placeholders
    battle_data = []
    W = np.zeros((num_arms, num_states, num_arms, num_states))
    if cfg.assisted_P:
        P_hat = P_true
        P_hat_baseline = P_true
    else:
        P_hat = np.ones((num_arms, num_states, num_states, num_actions)) / num_states
        P_hat_baseline = (
            np.ones((num_arms, num_states, num_states, num_actions)) / num_states
        )

    Z_sa = np.zeros((num_arms, num_states, num_actions))
    Z_sa_baseline = np.zeros((num_arms, num_states, num_actions))
    Z_sas = np.zeros((num_arms, num_states, num_states, num_actions))
    Z_sas_baseline = np.zeros((num_arms, num_states, num_states, num_actions))
    if cfg.reward_normalized:
        F_tilde = np.ones((num_arms, num_states, num_arms, num_states)) * (
            np.e / (np.e + 1)
        )
    else:
        F_tilde = np.ones((num_arms, num_states, num_arms, num_states)) * (1 - 1e-6)

    F_hat = np.ones((num_arms, num_states, num_arms, num_states)) * 0.5
    R_est = np.ones((num_arms, num_states)) * 0.5

    delta = np.ones((num_arms, num_states, num_actions))
    delta_baseline = np.ones((num_arms, num_states, num_actions))
    if cfg.assisted_P:
        delta = delta * 0.001
        delta_baseline = delta_baseline * 0.001
    conf = np.ones((num_arms, num_states, num_arms, num_states)) * np.sqrt(
        np.log(4 * num_states * num_actions * num_arms * K * env.H / (delta_coeff))
        / (2)
    )

    # Set self battles to 0.5
    for arm in range(num_arms):
        for state in range(num_states):
            F_hat[arm, state, arm, state] = 0.5
            conf[arm, state, arm, state] = 0.0
            F_tilde[arm, state, arm, state] = 0.5

    # Performance trackers
    train_curve = []
    baseline_curve = []
    rand_curve = []
    opt_curve = []
    dopl_train_regret = []
    baseline_train_regret = []
    dopl_train_regret_cumulative = []
    baseline_train_regret_cumulative = []
    # Loss trackers
    index_error_dopl = []
    index_error_baseline = []

    F_error = []
    P_error = []
    P_error_baseline = []
    Q_error = []
    R_est_error = []

    # Meta trackers
    episode_delta_tracker_min = []
    episode_delta_baseline_tracker_min = []
    episode_delta_tracker_mean = []
    episode_delta_baseline_tracker_mean = []
    episode_conf_tracker_mean = []
    episode_conf_tracker_min = []
    elp_cost_tracker = []
    elp_cost_baseline_tracker = []

    # Start training
    failure_point = K
    failure_point_baseline = K

    # Timing trackers
    dopl_timing_ticker = 0
    baseline_timing_ticker = 0

    for k in tqdm(range(K)):
        if cfg.reference_strategy == "random":
            ref_arm, ref_state = pick_random_ref(W)
        elif cfg.reference_strategy == "best":
            ref_arm, ref_state = pick_best_ref(W)
        elif cfg.reference_strategy == "fixed":
            ref_arm, ref_state = 0, 0
        ref_state = 0
        # Compute the corresponding index policy
        if cfg.enrich_F:
            # Use lemma in paper to estimte better F_tilde values
            start_time = time.time()
            F_tilde, F_hat = enrich_F(F_tilde, F_hat, (ref_arm, ref_state), conf)
            # Clip F_tilde and F_hat according to reward normalization
            if cfg.reward_normalized:
                F_tilde = np.clip(F_tilde, 1 / (np.e + 1), np.e / (np.e + 1))
                F_hat = np.clip(F_hat, 1 / (np.e + 1), np.e / (np.e + 1))
            else:
                F_tilde = np.clip(F_tilde, 1e-6, 1 - 1e-6)
                F_hat = np.clip(F_hat, 1e-6, 1 - 1e-6)
        Q_n_s = np.log(
            F_tilde[:, :, ref_arm, ref_state] / (1 - F_tilde[:, :, ref_arm, ref_state])
        )
        end_time = time.time()
        dopl_timing_ticker += end_time - start_time
        Q_true = np.log(
            F_true[:, :, ref_arm, ref_state] / (1 - F_true[:, :, ref_arm, ref_state])
        )
        ##compute the policy
        start_time = time.time()
        solution, elp_opt_cost = compute_ELP_pyomo(
            delta,
            P_hat,
            env.arm_constraint,
            num_states,
            num_actions,
            Q_n_s,
            num_arms,
        )
        end_time = time.time()
        dopl_timing_ticker += end_time - start_time
        _, elp_opt_cost_true = compute_ELP_pyomo(
            delta,
            P_hat,
            env.arm_constraint,
            num_states,
            num_actions,
            Q_true,
            num_arms,
        )
        elp_opt_cost_truly_true, _ = compute_optimal_pyomo(
            Q_true,
            env.arm_constraint,
            P_true,
            num_arms,
            num_states,
            num_actions,
        )
        ##compute the baseline policy
        start_time = time.time()
        solution_baseline, elp_opt_cost_baseline = compute_ELP_pyomo(
            delta_baseline,
            P_hat_baseline,
            env.arm_constraint,
            num_states,
            num_actions,
            R_est,
            num_arms,
        )
        end_time = time.time()
        baseline_timing_ticker += end_time - start_time
        start_time = time.time()
        if solution is not None:
            W_sas = solution
        else:
            elp_opt_cost = (
                elp_cost_tracker[-1] if len(elp_cost_tracker) > 0 else elp_opt_cost
            )
            if k == 0:
                print("No feasible solution in first iteration!")
                SystemExit()
            else:
                print("No feasible solution! Retaining the previous solution.")
                if failure_point == K:
                    failure_point = k
        elp_cost_tracker.append(elp_opt_cost)

        W_sa = np.sum(W_sas, axis=2)
        index_matrix_pre_nan = W_sa[:, :, 1] / (W_sa[:, :, 0] + W_sa[:, :, 1])
        index_matrix = np.nan_to_num(index_matrix_pre_nan, nan=0.0)
        end_time = time.time()
        dopl_timing_ticker += end_time - start_time
        start_time = time.time()
        if solution_baseline is not None:
            W_sas_baseline = solution_baseline
        else:
            elp_opt_cost_baseline = (
                elp_cost_baseline_tracker[-1]
                if len(elp_cost_baseline_tracker) > 0
                else elp_opt_cost_baseline
            )
            if k == 0:
                print("No feasible solution in first iteration!")
                SystemExit()
            else:
                print("No feasible solution! Retaining the previous solution.")
                if failure_point_baseline == K:
                    failure_point_baseline = k
        elp_cost_baseline_tracker.append(elp_opt_cost_baseline)
        W_sa_baseline = np.sum(W_sas_baseline, axis=2)
        index_matrix_baseline_pre_nan = W_sa_baseline[:, :, 1] / (
            W_sa_baseline[:, :, 0] + W_sa_baseline[:, :, 1]
        )
        index_matrix_baseline = np.nan_to_num(index_matrix_baseline_pre_nan, nan=0.0)
        end_time = time.time()
        baseline_timing_ticker += end_time - start_time
        # Evaluate the policy
        train_curve.append(eval(10, env, index_matrix))
        baseline_curve.append(eval(10, env, index_matrix_baseline))
        rand_curve.append(eval(10, env, np.random.rand(num_arms, num_states)))
        opt_curve.append(eval(10, env, env.opt_index))

        # Compute recosntruction losses
        index_error_dopl.append(np.linalg.norm(index_matrix - env.opt_index))
        index_error_baseline.append(
            np.linalg.norm(index_matrix_baseline - env.opt_index)
        )
        F_error.append(np.linalg.norm(F_hat - F_true))
        P_error.append(np.linalg.norm(P_hat - P_true))
        P_error_baseline.append(np.linalg.norm(P_hat_baseline - P_true))
        Q_error.append(np.linalg.norm(Q_n_s - Q_true))
        R_est_error.append(np.linalg.norm(R_est - R_true))

        # Update F_tilde according to alg 3
        # Meta trackers episode level
        delta_tracker = []
        delta_baseline_tracker = []
        conf_tracker = []
        # DOPL
        start_time = time.time()
        s_list = env.reset()
        dopl_train_regret_episode = 0
        for t in range(env.H):
            action = apply_index_policy(s_list, index_matrix, env.arm_constraint)
            s_dash_list, reward_dopl, _, _, info = env.step(action)
            dopl_train_regret_episode += reward_dopl
            for arm_id, s, a, s_dash in zip(
                range(num_arms), s_list, action, s_dash_list
            ):
                if not cfg.assisted_P:
                    Z_sa[arm_id, s, a] += 1
                    Z_sas[arm_id, s, s_dash, a] += 1
                    delta[arm_id, s, a] = np.sqrt(
                        np.log(
                            4
                            * num_states
                            * num_actions
                            * num_arms
                            * (k + 1)
                            * env.H
                            / (delta_coeff)
                        )
                        / (2 * Z_sa[arm_id, s, a])
                    )
                    delta_tracker.append(delta[arm_id, s, a])
                    P_hat[arm_id, s, s_dash, a] = Z_sas[
                        arm_id, s, s_dash, a
                    ] / np.maximum(1, Z_sa[arm_id, s, a])
                    true_diff = np.abs(
                        P_true[arm_id, s, s_dash, a] - P_hat[arm_id, s, s_dash, a]
                    )
                    if true_diff > delta[arm_id, s, a] or delta[arm_id, s, a] <= 0.0:
                        print(
                            f"Expected to fail ---- {true_diff} > {delta[arm_id, s, a]}"
                        )
                        SystemExit()
            for record in info["duelling_results"]:
                winner, loser = record
                W[winner, s_list[winner], loser, s_list[loser]] += 1
                battle_count = (
                    W[winner, s_list[winner], loser, s_list[loser]]
                    + W[loser, s_list[loser], winner, s_list[winner]]
                )
                F_hat[winner, s_list[winner], loser, s_list[loser]] = (
                    W[winner, s_list[winner], loser, s_list[loser]] / battle_count
                )
                F_hat[loser, s_list[loser], winner, s_list[winner]] = (
                    1 - F_hat[winner, s_list[winner], loser, s_list[loser]]
                )
                conf[winner, s_list[winner], loser, s_list[loser]] = conf[
                    loser, s_list[loser], winner, s_list[winner]
                ] = np.sqrt(
                    np.log(4 * num_states * num_arms * (k + 1) * env.H / conf_coeff)
                    / (2 * battle_count)
                )

                F_tilde[winner, s_list[winner], loser, s_list[loser]] = (
                    F_hat[winner, s_list[winner], loser, s_list[loser]]
                    + conf[winner, s_list[winner], loser, s_list[loser]]
                )
                F_tilde[loser, s_list[loser], winner, s_list[winner]] = (
                    F_hat[loser, s_list[loser], winner, s_list[winner]]
                    + conf[loser, s_list[loser], winner, s_list[winner]]
                )
                conf_tracker.append(conf[winner, s_list[winner], loser, s_list[loser]])
                if cfg.reward_normalized:
                    F_tilde = np.clip(
                        F_tilde, 1 / (np.e + 1), np.e / (np.e + 1)
                    )  # Since reward bounded between 0 and 1
                    F_hat = np.clip(F_hat, 1 / (np.e + 1), np.e / (np.e + 1))
                else:
                    F_tilde = np.clip(
                        F_tilde, 1e-6, 1 - 1e-6
                    )  # For numerical stability
                    F_hat = np.clip(F_hat, 1e-6, 1 - 1e-6)
            s_list = s_dash_list
        dopl_train_regret.append(env.opt_cost * env.H - dopl_train_regret_episode)
        dopl_train_regret_cumulative.append(sum(dopl_train_regret))
        episode_delta_tracker_min.append(min(delta_tracker))
        episode_delta_tracker_mean.append(np.mean(delta_tracker))
        episode_conf_tracker_min.append(min(conf_tracker))
        episode_conf_tracker_mean.append(np.mean(conf_tracker))
        end_time = time.time()
        dopl_timing_ticker += end_time - start_time
        # Baseline
        start_time = time.time()
        baseline_train_regret_episode = 0
        if not cfg.mle_cumulative:
            battle_data = []  # Flush prev battle_data
        s_list_baseline = env.reset()
        for t in range(env.H):
            action_baseline = apply_index_policy(
                s_list_baseline, index_matrix_baseline, env.arm_constraint
            )
            s_dash_list_baseline, reward_mle, _, _, info_baseline = env.step(
                action_baseline
            )
            baseline_train_regret_episode += reward_mle
            for arm_id, s, a, s_dash in zip(
                range(num_arms), s_list_baseline, action_baseline, s_dash_list_baseline
            ):
                if not cfg.assisted_P:
                    Z_sa_baseline[arm_id, s, a] += 1
                    Z_sas_baseline[arm_id, s, s_dash, a] += 1
                    delta_baseline[arm_id, s, a] = np.sqrt(
                        np.log(
                            4
                            * num_states
                            * num_actions
                            * num_arms
                            * (k + 1)
                            * env.H
                            / delta_coeff
                        )
                        / (2 * Z_sa_baseline[arm_id, s, a])
                    )
                    delta_baseline_tracker.append(delta_baseline[arm_id, s, a])
                    P_hat_baseline[arm_id, s, s_dash, a] = Z_sas_baseline[
                        arm_id, s, s_dash, a
                    ] / np.maximum(1, Z_sa_baseline[arm_id, s, a])
                    true_diff = np.abs(
                        P_true[arm_id, s, s_dash, a]
                        - P_hat_baseline[arm_id, s, s_dash, a]
                    )
                    if (
                        true_diff > delta_baseline[arm_id, s, a]
                        or delta_baseline[arm_id, s, a] <= 0.0
                    ):
                        print(
                            f"Expected to fail ---- {true_diff} > {delta_baseline[arm_id, s, a]}"
                        )
                        SystemExit()
            for record in info_baseline["duelling_results"]:
                winner, loser = record
                battle_data.append(
                    [winner, s_list_baseline[winner], loser, s_list_baseline[loser]]
                )
            s_list_baseline = s_dash_list_baseline
        baseline_train_regret.append(
            env.opt_cost * env.H - baseline_train_regret_episode
        )
        baseline_train_regret_cumulative.append(sum(baseline_train_regret))
        episode_delta_baseline_tracker_mean.append(np.mean(delta_baseline_tracker))
        episode_delta_baseline_tracker_min.append(min(delta_baseline_tracker))
        if cfg.mle_method == "first_order":
            R_est = mle_gradient_descent(np.array(battle_data), R_est)
        elif cfg.mle_method == "second_order":
            R_est = mle_bradley_terry(np.array(battle_data), R_est)
        end_time = time.time()
        baseline_timing_ticker += end_time - start_time
        wandb.log(
            {
                "k": k,
                "train_curve": train_curve[-1],
                "baseline_curve": baseline_curve[-1],
                "rand_curve": rand_curve[-1],
                "opt_curve": opt_curve[-1],
                "index_error_dopl": index_error_dopl[-1],
                "index_error_baseline": index_error_baseline[-1],
                "F_error": F_error[-1],
                "P_error": P_error[-1],
                "Q_error": Q_error[-1],
                "elp_cost_tracker": elp_cost_tracker[-1],
                "elp_cost_true": elp_opt_cost_true,
                "elp_cost_truly_true": elp_opt_cost_truly_true,
                "failure_point": failure_point,
                "failure_point_baseline": failure_point_baseline,
                "delta_min": episode_delta_tracker_min[-1],
                "delta_mean": episode_delta_tracker_mean[-1],
                "delta_baseline_min": episode_delta_baseline_tracker_min[-1],
                "delta_baseline_mean": episode_delta_baseline_tracker_mean[-1],
                "conf_min": episode_conf_tracker_min[-1],
                "conf_mean": episode_conf_tracker_mean[-1],
                "reference_arm": ref_arm,
                "reference_state": ref_state,
                "train_regret_dopl": dopl_train_regret[-1],
                "train_regret_baseline": baseline_train_regret[-1],
                "train_regret_cumulative_dopl": dopl_train_regret_cumulative[-1],
                "train_regret_cumulative_baseline": baseline_train_regret_cumulative[
                    -1
                ],
            }
        )
        # breakpoint()
    performance = {
        "train_curve": train_curve,
        "baseline_curve": baseline_curve,
        "rand_curve": rand_curve,
        "opt_curve": opt_curve,
        "dopl_train_regret": dopl_train_regret_cumulative,
        "baseline_train_regret": baseline_train_regret_cumulative,
    }
    loss = {
        "index_error_dopl": index_error_dopl,
        "index_error_baseline": index_error_baseline,
        "F_error": F_error,
        "P_error_dopl": P_error,
        "P_error_baseline": P_error_baseline,
        "Q_error": Q_error,
    }
    meta = {
        "delta_tracker_P": episode_delta_tracker_min,
        "delta_tracker_F": episode_conf_tracker_min,
        "elp_cost_tracker": elp_cost_tracker,
    }
    # breakpoint()
    print(f"DOPL timing: {dopl_timing_ticker}")
    print(f"Baseline timing: {baseline_timing_ticker}")
    # percentage increase from dopl to baseline
    print(
        f"Percentage increase in computation time: {(baseline_timing_ticker - dopl_timing_ticker) / dopl_timing_ticker * 100}"
    )
    return performance, loss, meta, failure_point


# Evaluate the policy
def eval(num_episodes, env, index_matrix):
    episode_rewards = []
    for ep_num in range(num_episodes):
        s_list = env.reset()
        episode_reward = 0
        for t in range(env.H):
            action = apply_index_policy(s_list, index_matrix, env.arm_constraint)
            s_dash_list, reward, _, _, _ = env.step(action)
            episode_reward += reward
            s_list = s_dash_list
        episode_rewards.append(episode_reward)
    return np.mean(episode_rewards)
