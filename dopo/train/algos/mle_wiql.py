import time
from tqdm import tqdm
import numpy as np
from dopo.utils import wandb_log_latest, normalize_matrix
from dopo.registry import register_training_function
from dopo.train.algos.mle_lp import mle_bradley_terry
from dopo.train.helpers import apply_index_policy


def a_seq(n):
    n += 1
    C = 1
    return C / np.ceil(n / 100)


@register_training_function("mle_wiql")
def train(env, cfg):
    start_time = time.time()
    K = cfg["K"]
    R_true = np.array(env.R_list)[:, :, 0]

    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    num_actions = env.R_list[0].shape[1]

    R_est = np.ones((num_arms, num_states)) * 0.5
    Q = np.zeros(
        (
            num_arms,
            num_states,
            num_actions,
        )
    )
    W = np.random.rand(num_arms, num_states)
    Z_sa = np.zeros((num_arms, num_states, num_actions))

    metrics = {
        "reward": [],
        "index_error": [],
        "R_error": [],
        "run_time": 0,
    }
    global_step = 0
    for k in tqdm(range(K)):
        # Start rollout using Q values as policy
        battle_data = []
        traj_states = []
        traj_actions = []
        traj_next_states = []
        s_list = env.reset()
        reward_episode = 0
        for t in range(env.H):
            global_step += 1
            epsilon = num_arms / (num_arms + global_step)
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
        for arm in range(num_arms):
            # Update Q values based on traj data
            for s, a, s_dash in zip(traj_states, traj_actions, traj_next_states):
                Q[arm, s[arm], a[arm]] = (1 - a_seq(Z_sa[arm, s[arm], a[arm]])) * Q[
                    arm, s[arm], a[arm]
                ] + a_seq(Z_sa[arm, s[arm], a[arm]]) * (
                    R_est[arm, s[arm]] + np.max(Q[arm, s_dash[arm], :])
                )
                # Update W value for state
                W[arm, s[arm]] = Q[arm, s[arm], 1] - Q[arm, s[arm], 0]
        metrics["R_error"].append(np.linalg.norm(R_est - R_true))
        metrics["index_error"].append(np.linalg.norm(W - env.opt_index))
        wandb_log_latest(metrics)
    end_time = time.time()
    metrics["run_time"] = end_time - start_time
    return metrics
