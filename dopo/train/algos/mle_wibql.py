from tqdm import tqdm
import numpy as np
from dopo.utils import wandb_log_latest
from dopo.registry import register_training_function
import torch
from dopo.train.algos.mle_lp import mle_bradley_terry
from dopo.train.helpers import apply_index_policy


def func(Q, num_states):
    """input : Q dimensionality [ arms, states, actions]
    input : num_states scalar (d in Borkar paper)
    output : func(Q) dimensionality [arms]"""
    return 1 / (2 * num_states) * np.sum(Q, axis=(1, 2))


@register_training_function("mle_wibql")
def train(env, cfg):
    K = cfg["K"]
    epsilon = cfg["epsilon"]
    R_true = np.array(env.R_list)[:, :, 0]

    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    num_actions = env.R_list[0].shape[1]

    R_est = np.ones((num_arms, num_states)) * 0.5
    Q = -np.ones(
        (
            num_arms,
            num_states,
            num_actions,
            num_states,
        )
    )
    W = np.zeros((num_arms, num_states))
    Z_sa = np.zeros((num_arms, num_states, num_actions))

    metrics = {"reward": [], "R_error": []}

    for k in tqdm(range(K)):
        # Start rollout using Q values as policy
        battle_data = []
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
                battle_data.append(
                    [winner, s_dash_list[winner], loser, s_dash_list[loser]]
                )
            s_list = s_dash_list

        metrics["reward"].append(reward_episode)
        # Update R_est using Bradley-Terry model
        R_est = mle_bradley_terry(np.array(battle_data), R_est)
        for state in range(num_states):  # k_hat in Borkar paper
            # Update Q values based on traj data
            for s, a, s_dash in zip(traj_states, traj_actions, traj_next_states):
                Q[np.arange(num_arms), s, a, state] = Q[
                    np.arange(num_arms), s, a, state
                ] + 1 / np.sqrt(Z_sa[np.arange(num_arms), s, a]) * (
                    (1 - a) * (R_est[np.arange(len(s)), s] + W[:, state])
                    + a * R_est[np.arange(len(s)), s]
                    + np.max(Q[np.arange(len(s_dash)), s_dash, :, state], axis=1)
                    - func(Q[:, :, :, state], num_states)
                    - Q[np.arange(num_arms), s, a, state]
                )
            # Update W value for state
            W[:, state] = W[:, state] + (1 / np.sqrt(k + 1)) * (
                Q[:, state, 1, state] - Q[:, state, 0, state]
            )
        metrics["R_error"].append(np.linalg.norm(R_est - R_true))
        wandb_log_latest(metrics, "mle_wibql")
    return metrics
