from tqdm import tqdm
from dopo.utils import wandb_log_latest
from dopo.registry import register_training_function
import random
import numpy as np
import time
from collections import deque
import markovianbandit as bandit
from scipy.stats import kendalltau


def normalize_P_hat(P_hat):
    num_arms, num_states, _, num_actions = P_hat.shape

    for arm_id in range(num_arms):
        for state in range(num_states):
            for action in range(num_actions):
                total_prob = np.sum(P_hat[arm_id, state, :, action])
                if total_prob > 0:
                    P_hat[arm_id, state, :, action] /= total_prob
                else:
                    # If the total probability is zero, initialize uniformly
                    P_hat[arm_id, state, :, action] = 1.0 / num_states
    return P_hat


def normalize_P_hat_sub(P_hat_sub):
    num_states, _, num_actions = P_hat_sub.shape

    for state in range(num_states):
        for action in range(num_actions):
            total_prob = np.sum(P_hat_sub[state, :, action])
            if total_prob > 0:
                P_hat_sub[state, :, action] /= total_prob
            else:
                P_hat_sub[state, :, action] = 1.0 / num_states
    return P_hat_sub


def ComputeLambdas(P_hat, rewards):
    lambdas = []
    P_hat = normalize_P_hat(P_hat)
    for arm_id in range(P_hat.shape[0]):
        P0 = P_hat[arm_id, :, :, 0]
        P1 = P_hat[arm_id, :, :, 1]
        R0 = rewards[arm_id]
        R1 = rewards[arm_id]
        model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
        lambdas.append(model.whittle_indices(discount=0.99))
    return np.array(lambdas)


def ComputeLambda(P_hat_sub, rewards_sub):
    lambda_sub = None
    P_hat_sub = normalize_P_hat_sub(P_hat_sub)
    P_0 = P_hat_sub[:, :, 0]
    P_1 = P_hat_sub[:, :, 1]
    R_0 = rewards_sub
    R_1 = rewards_sub
    model = bandit.restless_bandit_from_P0P1_R0R1(P_0, P_1, R_0, R_1)
    lambda_sub = model.whittle_indices(discount=0.99)
    return lambda_sub


def Subsample(arm_trajectory, n, num_states, num_actions):
    # Uniformly sample n tuples from arm_trajectory
    sampled_transitions = random.sample(arm_trajectory, n)

    P_hat_sub = np.zeros((num_states, num_states, num_actions))
    rewards_sub = np.zeros((num_states))
    # Count occurrences for transitions and rewards
    Z_s_sub = np.zeros(num_states)
    Z_sa_sub = np.zeros((num_states, num_actions))
    Z_sas_sub = np.zeros((num_states, num_states, num_actions))

    for s, a, r, s_dash in sampled_transitions:
        Z_s_sub[s] += 1
        Z_sa_sub[s, a] += 1
        Z_sas_sub[s, s_dash, a] += 1
        rewards_sub[s] += r

    # Normalize to get the transition probabilities and rewards
    for s in range(num_states):
        for a in range(num_actions):
            P_hat_sub[s, :, a] = Z_sas_sub[s, :, a] / np.maximum(1, Z_sa_sub[s, a])
        rewards_sub[s] /= np.maximum(1, Z_s_sub[s])

    return P_hat_sub, rewards_sub


def update_moving_average(current_avg, new_value, count):
    return (current_avg * (count - 1) + new_value) / count


@register_training_function("sub_sample_gast")
def train(env, cfg):
    start_time = time.time()
    K = cfg["K"]
    history_size = cfg["history_size"]

    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    num_actions = env.R_list[0].shape[1]

    P_true = np.array(env.P_list)

    W = np.random.rand(num_arms, num_states)  # Global whittle index estimates

    #  Global transition kernel estimate
    P_hat = np.ones((num_arms, num_states, num_states, num_actions)) / num_states
    Z_s = np.zeros(
        (num_arms, num_states), dtype=int
    )  # (N_k_t in paper; num_sub_samples)
    Z_sa = np.zeros((num_arms, num_states, num_actions), dtype=int)
    Z_sas = np.zeros((num_arms, num_states, num_states, num_actions), dtype=int)

    # Transitions buffer for subsampling kernel estimate
    transitions = {
        arm_id: deque(maxlen=cfg["history_size"]) for arm_id in range(num_arms)
    }
    # Rewards buffer for subsampling reward estimate
    reward_averages = np.zeros((num_arms, num_states))

    metrics = {
        "reward": [],
        "P_error": [],
        "leader_set_error_subsampling": [],
        "run_time": 0,
    }

    # Initial exploration phase
    exploration_steps_per_s_a = 1
    env.init_exploration_phase = True
    for _ in range(exploration_steps_per_s_a):
        for state in range(num_states):
            for action in range(num_actions):
                states_list = [state] * num_arms
                env.set_state(states_list)
                actions_list = np.array([action] * num_arms, dtype=int)
                s_dash_list, reward, _, _, info = env.step(actions_list)
                rewards_list = info["arm_rewards"]
                for arm_id, s_dash in zip(range(num_arms), s_dash_list):
                    reward_averages[arm_id, state] = update_moving_average(
                        reward_averages[arm_id, state],
                        rewards_list[arm_id],
                        Z_s[arm_id, state] + 1,
                    )
                    Z_s[arm_id, state] += 1
                    transitions[arm_id].append(
                        (state, action, rewards_list[arm_id], s_dash)
                    )
                    # Update global P_hat estimate
                    Z_sa[arm_id, state, action] += 1
                    Z_sas[arm_id, state, s_dash, action] += 1

    env.init_exploration_phase = False

    for k in tqdm(range(K)):

        s_list = env.reset()
        reward_episode = 0
        for t in range(env.H):
            # Global estimates
            global_reward_est = reward_averages
            P_hat = Z_sas / np.maximum(
                1, np.repeat(np.expand_dims(Z_sa, axis=2), repeats=num_states, axis=2)
            )
            W = ComputeLambdas(P_hat, global_reward_est)

            # Compute leader set
            current_indices = W[np.arange(num_arms), s_list]
            leader_set = np.argsort(current_indices)[-env.arm_constraint :]

            # True leader set
            true_indices = env.whittle_indices[np.arange(num_arms), s_list]
            true_leader_set = np.argsort(true_indices)[-env.arm_constraint :]

            # Conduct (N - arm_constraint) sub sampling battles
            for contender_arm in range(num_arms):
                if contender_arm in leader_set:
                    continue
                n_contender = min(int(np.min(Z_s[contender_arm][:])), history_size)
                for leader_arm in range(num_arms):
                    if leader_arm not in leader_set:
                        continue
                    n_leader = min(int(np.min(Z_s[leader_arm][:])), history_size)
                    if n_leader >= n_contender:  # Subsample leader arm
                        P_hat_sub, rewards_sub = Subsample(
                            transitions[leader_arm],
                            n_contender,
                            num_states,
                            num_actions,
                        )

                        W_sub = ComputeLambda(P_hat_sub, rewards_sub)
                        if (
                            W_sub[s_list[leader_arm]]
                            < W[contender_arm, s_list[contender_arm]]
                        ):  # Remove weakest leader and replace with contender but in correct ranking order

                            # Find the position of the current leader_arm in the leader_set
                            leader_arm_index = np.where(leader_set == leader_arm)[0][0]

                            # Insert contender_arm right above leader_arm
                            leader_set = np.insert(
                                leader_set, leader_arm_index + 1, contender_arm
                            )

                            # Remove the weakest leader, which is the first element in leader_set
                            leader_set = leader_set[1:]
                    else:  # Subsample contender arm
                        P_hat_sub, rewards_sub = Subsample(
                            transitions[contender_arm],
                            n_leader,
                            num_states,
                            num_actions,
                        )

                        W_sub = ComputeLambda(P_hat_sub, rewards_sub)
                        if (
                            W_sub[s_list[contender_arm]]
                            > W[leader_arm, s_list[leader_arm]]
                        ):
                            # Find the position of the current leader_arm in the leader_set
                            leader_arm_index = np.where(leader_set == leader_arm)[0][0]

                            # Insert contender_arm right above leader_arm
                            leader_set = np.insert(
                                leader_set, leader_arm_index + 1, contender_arm
                            )

                            # Remove the weakest leader, which is the first element in leader_set
                            leader_set = leader_set[1:]

            action = np.zeros(num_arms, dtype=int)
            action[leader_set] = 1
            s_dash_list, reward, _, _, info = env.step(action)
            reward_episode += reward
            rewards_list = info["arm_rewards"]
            # Update P estimate
            for arm_id, s, a, s_dash in zip(
                range(num_arms), s_list, action, s_dash_list
            ):
                Z_s[arm_id, s] += 1
                reward_averages[arm_id, s] = update_moving_average(
                    reward_averages[arm_id, s], rewards_list[arm_id], Z_s[arm_id, s]
                )
                # Update global P_hat estimate
                Z_sa[arm_id, s, a] += 1
                Z_sas[arm_id, s, s_dash, a] += 1
                transitions[arm_id].append((s, a, rewards_list[arm_id], s_dash))

            # Compute leader set error
            set_difference_size = len(set(true_leader_set) - set(leader_set))
            metrics["leader_set_error_subsampling"].append(set_difference_size)
            s_list = s_dash_list
        if np.isinf(W).any() or np.isnan(W).any():
            print(
                "W contains infinity or nan values. This may indicate numerical instability in the algorithm."
            )  # Warning

        metrics["P_error"].append(np.linalg.norm(P_hat - P_true))
        metrics["reward"].append(reward_episode)
        wandb_log_latest(metrics)
    end_time = time.time()
    metrics["run_time"] = end_time - start_time
    return metrics
