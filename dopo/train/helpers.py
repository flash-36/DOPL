import numpy as np
from dopo.utils import compute_optimal


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


def get_opt_performance(env):
    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    num_actions = env.R_list[0].shape[1]
    # Compute opt objective for problem
    r_n_s = np.array(env.R_list)[:, :, 0]
    optimal_cost, opt_occupancy = compute_optimal(
        r_n_s,
        env.arm_constraint,
        np.array(env.P_list),
        num_arms,
        num_states,
        num_actions,
    )
    index_matrix_true = opt_occupancy[:, :, 1] / (
        opt_occupancy[:, :, 0] + opt_occupancy[:, :, 1]
    )
    return optimal_cost * env.T, index_matrix_true


def compute_F_true(env):
    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    F_true = np.zeros((num_arms, num_states, num_arms, num_states))
    for i in range(num_arms):
        for s_i in range(num_states):
            for j in range(num_arms):
                for s_j in range(num_states):
                    F_true[i, s_i, j, s_j] = np.exp(env.R_list[i][s_i, 0]) / (
                        np.exp(env.R_list[i][s_i, 0]) + np.exp(env.R_list[j][s_j, 0])
                    )
    return F_true
