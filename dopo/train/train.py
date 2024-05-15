import numpy as np
from tqdm import tqdm
from dopo.utils import compute_ELP, compute_ELP_pyomo
from dopo.train.helpers import apply_index_policy, compute_F_true
import logging

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
    return ref_arm, ref_state


def pick_random_ref(W):
    ref_arm = np.random.randint(0, W.shape[0])
    ref_state = np.random.randint(0, W.shape[1])
    return ref_arm, ref_state


def train(env, cfg):
    FirstTimeWarning = True
    # Extract training parameters
    K = cfg["K"]
    H = cfg["H"]
    delta_coeff = cfg["delta_coeff"]
    conf_coeff = cfg["conf_coeff"]

    # Initialize utility variables
    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    num_actions = env.R_list[0].shape[1]
    # Initialze placeholders
    W = np.zeros((num_arms, num_states, num_arms, num_states))
    P_hat = np.ones((num_arms, num_states, num_states, num_actions)) / num_states
    Z_sa = np.zeros((num_arms, num_states, num_actions))
    Z_sas = np.zeros((num_arms, num_states, num_states, num_actions))

    F_tilde = np.ones((num_arms, num_states, num_arms, num_states)) * 0.7311
    F_hat = np.ones((num_arms, num_states, num_arms, num_states)) * 0.7311
    delta = np.ones((num_arms, num_states, num_actions))

    # Performance trackers
    train_curve = []
    rand_curve = []
    opt_curve = []
    # Loss trackers
    index_error = []
    F_error = []
    P_error = []

    # True values
    P_true = np.array(env.P_list)
    F_true = compute_F_true(env)
    # Trackers for debugging
    confidence_F = []
    confidence_P = []
    # Start training
    failure_point = K
    for k in tqdm(range(K)):
        ref_arm, ref_state = pick_random_ref(W)
        # ref_arm, ref_state = pick_best_ref(W)
        # check_stochasticity(P_hat)
        # Compute the corresponding index policy
        # breakpoint()
        Q_n_s = -np.log(F_tilde[ref_arm][ref_state] / (1 - F_tilde[ref_arm][ref_state]))
        Q_true = -np.log(F_true[ref_arm][ref_state] / (1 - F_true[ref_arm][ref_state]))
        print(Q_n_s)
        print(Q_true)
        if confidence_F:
            print(confidence_F[-1])
        ##compute the policy
        solution = compute_ELP_pyomo(
            delta,
            P_hat,
            env.arm_constraint,
            num_states,
            num_actions,
            Q_n_s,
            num_arms,
        )
        if solution is not None:
            W_sas = solution
        else:
            if k == 0:
                print("No feasible solution in first iteration!")
            else:
                print("No feasible solution! Retaining the previous solution.")
                if failure_point == K:
                    failure_point = k
        W_sa = np.sum(W_sas, axis=2)
        index_matrix = W_sa[:, :, 1] / (W_sa[:, :, 0] + W_sa[:, :, 1])
        index_matrix = np.nan_to_num(index_matrix, nan=0.0)

        # Evaluate the policy
        train_curve.append(eval(10, env, index_matrix))
        rand_curve.append(eval(10, env, np.random.rand(num_arms, num_states)))
        opt_curve.append(eval(10, env, env.opt_index))

        # Compute recosntruction losses
        index_error.append(np.linalg.norm(index_matrix - env.opt_index))
        F_error.append(np.linalg.norm(F_hat - F_true))
        P_error.append(np.linalg.norm(P_hat - P_true))

        # Update F_tilde according to alg 2
        for h in range(H):
            s_list = env.reset()
            for t in range(env.T):
                action = apply_index_policy(s_list, index_matrix, env.arm_constraint)
                s_dash_list, reward, _, _, info = env.step(action)
                for arm_id, s, a, s_dash in zip(
                    range(num_arms), s_list, action, s_dash_list
                ):
                    Z_sa[arm_id, s, a] += 1
                    Z_sas[arm_id, s, s_dash, a] += 1
                    delta[arm_id, s, a] = np.sqrt(
                        np.log(
                            4
                            * num_states
                            * num_actions
                            * num_arms
                            * (k + 1)
                            * H
                            * env.T
                            / (delta_coeff)
                        )
                        / (2 * Z_sa[arm_id, s, a])
                    )
                    confidence_P.append(delta[arm_id, s, a])
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
                    ## Ujwal modification of algorithm #####
                    F_hat[loser, s_list[loser], winner, s_list[winner]] = (
                        1 - F_hat[winner, s_list[winner], loser, s_list[loser]]
                    )
                    ########################################
                    F_tilde[winner, s_list[winner], loser, s_list[loser]] = F_hat[
                        winner, s_list[winner], loser, s_list[loser]
                    ] + np.sqrt(
                        np.log(
                            4 * num_states * num_arms * (k + 1) * H * env.T / conf_coeff
                        )
                        / (2 * battle_count)
                    )

                    ## Ujwal modification of algorithm #####
                    F_tilde[loser, s_list[loser], winner, s_list[winner]] = F_hat[
                        loser, s_list[loser], winner, s_list[winner]
                    ] + np.sqrt(
                        np.log(
                            4 * num_states * num_arms * (k + 1) * H * env.T / conf_coeff
                        )
                        / (2 * battle_count)
                    )
                    confidence_F.append(
                        np.sqrt(
                            np.log(
                                4
                                * num_states
                                * num_arms
                                * (k + 1)
                                * H
                                * env.T
                                / conf_coeff
                            )
                            / (2 * battle_count)
                        )
                    )
                    F_tilde = np.clip(F_tilde, 0.2689, 0.7311)
                    ########################################
                    # F_tilde = np.clip(
                    #     F_tilde, 1e-6, 1 - 1e-6
                    # )  # For numerical stability
                s_list = s_dash_list

    performance = {
        "train_curve": train_curve,
        "rand_curve": rand_curve,
        "opt_curve": opt_curve,
    }
    loss = {"index_error": index_error, "F_error": F_error, "P_error": P_error}
    # breakpoint()
    return performance, loss, failure_point


# Evaluate the policy
def eval(num_episodes, env, index_matrix):
    episode_rewards = []
    for ep_num in range(num_episodes):
        s_list = env.reset()
        episode_reward = 0
        for t in range(env.T):
            action = apply_index_policy(s_list, index_matrix, env.arm_constraint)
            s_dash_list, reward, _, _, _ = env.step(action)
            episode_reward += reward
            s_list = s_dash_list
        episode_rewards.append(episode_reward)
    return np.mean(episode_rewards)
