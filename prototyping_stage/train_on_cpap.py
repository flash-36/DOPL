import numpy as np
from dopo.envs import MultiArmRestlessDuellingEnv

from dopo.utils import load_arm, ELP

import matplotlib.pyplot as plt

H = 20
K = 100
T = 10
delta_coeff = 0.1
conf_coeff = 0.1


def main():
    # Load the transition and reward matrices for both arm types
    P1, R1 = load_arm("cpap_arm_type_1")
    P2, R2 = load_arm("cpap_arm_type_2")

    # Initialize environment with 5 arms of each type
    P_list = [P1] * 2 + [P2] * 2
    R_list = [R1] * 2 + [R2] * 2
    arm_constraint = 2  # Define how many arms can be pulled at once

    # Create an instance of the duelling environment
    env = MultiArmRestlessDuellingEnv(arm_constraint, P_list, R_list)

    train(env)


def eval(env, index_matrix):
    s_list = env.reset()
    total_reward = 0
    for t in range(T):
        action = apply_index_policy(s_list, index_matrix)
        s_dash_list, reward, _, _, _ = env.step(action)
        total_reward += reward
    return total_reward


def apply_index_policy(state_list, index_matrix):
    """
    state_list: list of states for each arm; 1*num_arms
    index_matrix: num_arms*num_states

    return: list of actions for each arm; 1*num_arms
    """
    index_vector = index_matrix[np.arange(len(state_list)), state_list]


def train(env):

    # Initialze placeholders
    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    num_actions = env.R_list[0].shape[1]
    W = np.zeros((num_arms, num_states, num_arms, num_states))
    C = np.zeros((num_arms, num_states, num_arms, num_states))
    P_hat = np.ones((num_arms, num_states, num_states, num_actions)) / num_states
    Z_sa = np.zeros((num_arms, num_states, num_actions))
    Z_sas = np.zeros((num_arms, num_states, num_actions, num_states))
    train_curve = []
    F_tilde = np.zeros((num_arms, num_states, num_arms, num_states)) * 0.5
    F_hat = np.zeros((num_arms, num_states, num_arms, num_states)) * 0.5
    delta = np.ones((num_arms, num_states, num_actions)) * delta_coeff
    ref_arm, ref_state = 0, 0

    # Start traing
    for k in range(K):
        # Compute the corresponding index policy
        ## Compte Q_n_s
        Q_n_s = np.log((1 - F_tilde[ref_arm][ref_state]) / F_tilde[ref_arm][ref_state])
        Q_n_s = np.zeros_like(Q_n_s)
        Q_n_s[ref_arm][ref_state] = 20
        ##compute the policy
        e = ELP(
            delta, P_hat, env.arm_constraint, num_states, num_actions, Q_n_s, num_arms
        )  # TODO: Can ELP handle infs
        W_sas = e.compute_ELP()
        W_sa = np.sum(W_sas, axis=2)
        index_matrix = W_sa[:, :, 1] / (W_sa[:, :, 0] + W_sa[:, :, 1])

        # Evaluate the policy
        train_curve.append(eval(env, index_matrix))

        # Update F_tilde according to alg 2
        for h in range(H):
            s_list = env.reset()
            for t in range(T):
                # action = policy(s_list)
                action = np.array([1, 1, 0, 0])
                s_dash_list, reward, _, _, info = env.step(action)
                for arm_id, s, a, s_dash in zip(
                    range(num_arms), s_list, action, s_dash_list
                ):
                    Z_sa[arm_id, s, a] += 1
                    Z_sas[arm_id, s, a, s_dash] += 1
                for record in info["duelling_results"]:
                    winner, loser = record
                    W[winner, s_list[winner], loser, s_list[loser]] += 1
                    battle_count = (
                        W[winner, s_list[winner], loser, s_list[loser]]
                        + W[loser, s_list[loser], winner, s_list[winner]]
                    )
                    F_hat[winner, s_list[winner], loser, s_list[loser]] = (
                        W[winner, s_list[winner], loser, s_list[loser]] / battle_count
                    )  # TODO: Check if the indentation level is correct
                    F_tilde[winner, s_list[winner], loser, s_list[loser]] = F_hat[
                        winner, s_list[winner], loser, s_list[loser]
                    ] + conf_coeff * np.sqrt(
                        1 / battle_count
                    )  # Make this closer to the actual formula?

        # Construct set of plausible transition kernels (i.e compute its center and radii)
        P_hat = Z_sas / np.maximum(Z_sa[:, :, :, np.newaxis], 1)
        delta = delta_coeff * np.sqrt(
            1 / Z_sa
        )  # Make this closer to the actual formula?
        breakpoint()

    # plot trainin curve
    plt.plot(train_curve)


if __name__ == "__main__":
    main()
