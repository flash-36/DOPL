from tqdm import tqdm
import numpy as np
from dopo.utils import wandb_log_latest
from dopo.registry import register_training_function
from dopo.train.algos.mle_lp import mle_bradley_terry
from dopo.train.helpers import (
    apply_index_policy,
    compute_F_true,
    enrich_F,
    pick_best_ref,
)


def func(Q, num_states):
    """input : Q dimensionality [ states, actions]
    input : num_states scalar (d in Borkar paper)
    output : func(Q) scalar"""
    return 1 / (2 * num_states) * np.sum(Q)


def a_seq(n):
    n += 1
    C = 100
    return C / np.ceil(n / 500)


def b_seq(n):
    n += 1
    C_dash = 100
    N = 2
    return C_dash / (1 + np.ceil(n * np.log(n) / 500)) if n % N == 0 else 0


@register_training_function("direct_wibql")
def train(env, cfg):
    K = cfg["K"]
    epsilon = cfg["epsilon"]
    eps = cfg["eps"]
    R_true = np.array(env.R_list)[:, :, 0]
    F_true = compute_F_true(env)

    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    num_actions = env.R_list[0].shape[1]

    Q = np.zeros(
        (
            num_arms,
            num_states,
            num_actions,
            num_states,
        )
    )
    Q[:, :, 1, :] = 1
    W = np.zeros((num_arms, num_states))
    Wins = np.zeros((num_arms, num_states, num_arms, num_states))
    Z_sa = np.zeros((num_arms, num_states, num_actions))
    F_tilde = np.ones((num_arms, num_states, num_arms, num_states)) * (
        np.e / (np.e + 1) if cfg.reward_normalized else 1 - 1e-6
    )
    F_hat = np.ones((num_arms, num_states, num_arms, num_states)) * 0.5
    conf = np.ones((num_arms, num_states, num_arms, num_states)) * np.sqrt(
        np.log(4 * num_states * num_actions * num_arms * K * env.H / epsilon) / 2
    )

    for arm in range(num_arms):
        for state in range(num_states):
            F_hat[arm, state, arm, state] = 0.5
            conf[arm, state, arm, state] = 0.0
            F_tilde[arm, state, arm, state] = 0.5

    metrics = {"reward": [], "F_error": [], "R_error": []}

    for k in tqdm(range(K)):
        # Start rollout using Q values as policy
        traj_states = []
        traj_actions = []
        traj_next_states = []
        s_list = env.reset()
        reward_episode = 0
        for t in range(env.H):
            if np.random.rand() < epsilon:
                action = apply_index_policy(
                    s_list, np.random.rand(num_arms, num_states), env.arm_constraint
                )
            else:
                action = apply_index_policy(s_list, W, env.arm_constraint)
            s_dash_list, reward, _, _, info = env.step(action)
            reward_episode += reward
            traj_states.append(s_list)
            traj_actions.append(action)
            traj_next_states.append(s_dash_list)
            for arm_id, s, a, s_dash in zip(
                range(num_arms), s_list, action, s_dash_list
            ):
                Z_sa[arm_id, s, a] += 1
            for record in info["duelling_results"]:
                winner, loser = record
                Wins[winner, s_list[winner], loser, s_list[loser]] += 1
                battle_count = (
                    Wins[winner, s_list[winner], loser, s_list[loser]]
                    + Wins[loser, s_list[loser], winner, s_list[winner]]
                )
                F_hat[winner, s_list[winner], loser, s_list[loser]] = (
                    Wins[winner, s_list[winner], loser, s_list[loser]] / battle_count
                )
                F_hat[loser, s_list[loser], winner, s_list[winner]] = (
                    1 - F_hat[winner, s_list[winner], loser, s_list[loser]]
                )
                conf[winner, s_list[winner], loser, s_list[loser]] = conf[
                    loser, s_list[loser], winner, s_list[winner]
                ] = np.sqrt(
                    np.log(4 * num_states * num_arms * (k + 1) * env.H / eps)
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

        metrics["reward"].append(reward_episode)

        # Get R_est using F_tilde
        ref_arm, ref_state = pick_best_ref(Wins)
        F_tilde, F_hat = enrich_F(F_tilde, F_hat, (ref_arm, ref_state), conf)
        if cfg.reward_normalized:
            F_tilde = np.clip(F_tilde, 1 / (np.e + 1), np.e / (np.e + 1))
            F_hat = np.clip(F_hat, 1 / (np.e + 1), np.e / (np.e + 1))
        else:
            F_tilde = np.clip(F_tilde, 1e-6, 1 - 1e-6)
            F_hat = np.clip(F_hat, 1e-6, 1 - 1e-6)

        # Use F estimate to compute index (policy)
        R_est = np.log(
            F_tilde[:, :, ref_arm, ref_state] / (1 - F_tilde[:, :, ref_arm, ref_state])
        )

        for arm in range(num_arms):
            for state in range(num_states):  # k_hat in Borkar paper
                # Update Q values based on traj data
                for s, a, s_dash in zip(traj_states, traj_actions, traj_next_states):
                    Q[arm, s[arm], a[arm], state] = Q[
                        arm, s[arm], a[arm], state
                    ] + a_seq(Z_sa[arm, s[arm], a[arm]]) * (
                        (1 - a[arm]) * (R_est[arm, s[arm]] + W[arm, state])
                        + a[arm] * R_est[arm, s[arm]]
                        + np.max(Q[arm, s_dash[arm], :, state])
                        - func(Q[arm, :, :, state], num_states)
                        - Q[arm, s[arm], a[arm], state]
                    )
                # Update W value for state
                W[arm, state] = W[arm, state] + (b_seq(k)) * (
                    Q[arm, state, 1, state] - Q[arm, state, 0, state]
                )
        metrics["R_error"].append(np.linalg.norm(R_est - R_true))
        metrics["F_error"].append(np.linalg.norm(F_tilde - F_true))
        wandb_log_latest(metrics)
    return metrics
