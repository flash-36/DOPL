from tqdm import tqdm
import numpy as np
from dopo.utils import wandb_log_latest, normalize_matrix
from dopo.registry import register_training_function
from dopo.train.algos.mle_lp import mle_bradley_terry
from dopo.train.helpers import apply_index_policy
import time


def func(Q, num_states):
    """input : Q dimensionality [ states, actions]
    input : num_states scalar (d in Borkar paper)
    output : func(Q) scalar"""
    return 1 / (2 * num_states) * np.sum(Q)


def a_seq(n, step_size_params):
    n += 1
    C = step_size_params["C"]
    D = step_size_params["D"]
    return C / np.ceil(n / D)


def b_seq(n, step_size_params):
    n += 1
    C_dash = step_size_params["C_dash"]
    D_dash = step_size_params["D_dash"]
    return C_dash / (1 + np.ceil(n * np.log(n) / D_dash))


@register_training_function("wibql_fg")
def train(env, cfg):
    start_time = time.time()
    K = cfg["K"]
    epsilon = cfg["epsilon"]
    R_true = np.array(env.R_list)[:, :, 0]

    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    num_actions = env.R_list[0].shape[1]

    Q = np.random.rand(
        num_arms,
        num_states,
        num_actions,
        num_states,
    )
    W = np.random.rand(num_arms, num_states)
    Z_sa = np.zeros((num_arms, num_states, num_actions))

    metrics = {"reward": [], "index_error": [], "run_time": 0}

    for k in tqdm(range(K)):
        # Start rollout using Q values as policy
        # battle_data = []
        # traj_states = []
        # traj_actions = []
        # traj_rewards = []
        # traj_next_states = []
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
            rewards_list = info["arm_rewards"]
            reward_episode += reward
            # traj_states.append(s_list)
            # traj_actions.append(action)
            # traj_rewards.append(rewards_list)
            # traj_next_states.append(s_dash_list)
            for arm_id, s, a, r, s_dash in zip(
                range(num_arms), s_list, action, rewards_list, s_dash_list
            ):
                Z_sa[arm_id, s, a] += 1
                for state in range(num_states): # k_hat in Borkar paper
                    # Update Q values
                    Q[arm_id, s, a, state] = Q[arm_id, s, a, state] + a_seq(Z_sa[arm_id, s, a], cfg["step_size_params"]) * (
                        (1 - a) * (r + W[arm_id, state])
                        + a * r
                        + np.max(Q[arm_id, s_dash, :, state])
                        - func(Q[arm_id, :, :, state], num_states)
                        - Q[arm_id, s, a, state]
                    )
                    # Update W value for state
                    W[arm_id, state] = W[arm_id, state] + (b_seq(k*env.H + t,cfg["step_size_params"])) * (
                        Q[arm_id, state, 1, state] - Q[arm_id, state, 0, state]
                    )
            s_list = s_dash_list
        # Check for infinities in W
        if np.isinf(W).any() or np.isnan(W).any():
            raise ValueError("W contains infinity values. This may indicate numerical instability in the algorithm.")

        metrics["reward"].append(reward_episode)
        
        # for arm in range(num_arms):
        #     for state in range(num_states):  # k_hat in Borkar paper
        #         # Update Q values based on traj data
        #         for s, a, r, s_dash in zip(traj_states, traj_actions, traj_rewards, traj_next_states):
        #             Q[arm, s[arm], a[arm], state] = Q[
        #                 arm, s[arm], a[arm], state
        #             ] + a_seq(Z_sa[arm, s[arm], a[arm]], cfg["step_size_params"]) * (
        #                 (1 - a[arm]) * (r[arm] + W[arm, state])
        #                 + a[arm] * r[arm]
        #                 + np.max(Q[arm, s_dash[arm], :, state])
        #                 - func(Q[arm, :, :, state], num_states)
        #                 - Q[arm, s[arm], a[arm], state]
        #             )
        #         # Update W value for state
        #         W[arm, state] = W[arm, state] + (b_seq(k,cfg["step_size_params"])) * (
        #             Q[arm, state, 1, state] - Q[arm, state, 0, state]
        #         )
        metrics["index_error"].append(np.linalg.norm(W - env.opt_index))
        wandb_log_latest(metrics)
    end_time = time.time()
    metrics["run_time"] = end_time - start_time
    return metrics
