import numpy as np
from dopo.envs import MultiArmRestlessDuellingEnv

from dopo.utils import load_arm, ELP, Optimal

import matplotlib.pyplot as plt

H = 20
K = 100
T = 10
delta_coeff = 1
conf_coeff = 0.1


def main():
    # Load the transition and reward matrices for both arm types
    P1, R1 = load_arm("cpap_arm_type_1")
    P2, R2 = load_arm("cpap_arm_type_2")

    # Initialize environment with 5 arms of each type
    P_list = [P1] * 5 + [P2] * 5
    R_list = [R1] * 5 + [R2] * 5
    arm_constraint = 2  # Define how many arms can be pulled at once

    # Create an instance of the duelling environment
    env = MultiArmRestlessDuellingEnv(arm_constraint, P_list, R_list)

    train(env)


def eval(env, index_matrix):
    s_list = env.reset()
    total_reward = 0
    for t in range(T):
        action = apply_index_policy(s_list, index_matrix, env.arm_constraint)
        s_dash_list, reward, _, _, _ = env.step(action)
        total_reward += reward
    return total_reward


def apply_index_policy(state_list, index_matrix, arm_constraint):
    """
    state_list: list of states for each arm; 1*num_arms
    index_matrix: num_arms*num_states

    return: list of actions for each arm; 1*num_arms
    """
    index_vector = index_matrix[np.arange(len(state_list)), state_list]
    largest_indices = np.argsort(index_vector)[-arm_constraint:]
    action = np.zeros(len(index_vector), dtype=int)
    action[largest_indices] = 1
    return action


def train(env):
    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    num_actions = env.R_list[0].shape[1]
    ref_arm, ref_state = 1, 2
    # Compute opt objective for problem self, reward, budget, P, n_arms, n_state, n_action
    r_n_s = np.array(env.R_list)[:, :, 0]
    opt = Optimal(
        r_n_s,
        env.arm_constraint,
        np.array(env.P_list),
        num_arms,
        num_states,
        num_actions,
    )
    optimal_cost = opt.compute_optimal() * T
    # Initialze placeholders

    W = np.zeros((num_arms, num_states, num_arms, num_states))
    P_hat = np.ones((num_arms, num_states, num_states, num_actions)) / num_states
    Z_sa = np.zeros((num_arms, num_states, num_actions))
    Z_sas = np.zeros((num_arms, num_states, num_states, num_actions))
    train_curve = []
    F_tilde = np.ones((num_arms, num_states, num_arms, num_states)) * 0.5
    F_hat = np.ones((num_arms, num_states, num_arms, num_states)) * 0.5
    delta = np.ones((num_arms, num_states, num_actions)) * delta_coeff

    # Start traing
    for k in range(K):
        # Compute the corresponding index policy
        ## Compte Q_n_s
        # print(F_tilde)
        # print("+" * 40)
        Q_n_s = np.log((1 - F_tilde[ref_arm][ref_state]) / F_tilde[ref_arm][ref_state])
        # print(Q_n_s)
        # Q_n_s = np.zeros_like(Q_n_s)
        # Q_n_s[ref_arm][ref_state] = 20
        ##compute the policy
        # breakpoint()
        e = ELP(
            delta, P_hat, env.arm_constraint, num_states, num_actions, Q_n_s, num_arms
        )  # TODO: Can ELP handle infs
        # breakpoint()
        W_sas = e.compute_ELP()
        # W_sas = np.ones((num_arms, num_states, num_states, num_actions))
        # W_sas = np.random.dirichlet(np.ones(W_sas.size)).reshape(W_sas.shape)
        W_sa = np.sum(W_sas, axis=2)
        index_matrix = W_sa[:, :, 1] / (W_sa[:, :, 0] + W_sa[:, :, 1])
        # print(W_sa)
        # breakpoint()

        # Evaluate the policy
        train_curve.append(eval(env, index_matrix))

        # Update F_tilde according to alg 2
        for h in range(H):
            s_list = env.reset()
            for t in range(T):
                action = apply_index_policy(s_list, index_matrix, env.arm_constraint)
                # action = np.array([1, 1, 0, 0])
                s_dash_list, reward, _, _, info = env.step(action)
                for arm_id, s, a, s_dash in zip(
                    range(num_arms), s_list, action, s_dash_list
                ):
                    # breakpoint()
                    Z_sa[arm_id, s, a] += 1
                    Z_sas[arm_id, s, s_dash, a] += 1
                    delta[arm_id, s, a] = (delta_coeff) * np.sqrt(
                        1 / (2 * max(1, (Z_sa[arm_id, s, a])))
                    )
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
        P_hat = Z_sas / np.maximum(Z_sa[:, :, np.newaxis, :], 1)

    # plot trainin curve
    opt_curve = [optimal_cost] * K
    plt.plot(train_curve)
    plt.plot(opt_curve)
    plt.savefig("train_curve.png")


if __name__ == "__main__":
    main()
