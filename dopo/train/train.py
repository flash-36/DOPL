import numpy as np
from tqdm import tqdm
from dopo.utils import compute_ELP, check_stochasticity
from dopo.train.helpers import apply_index_policy, compute_F_true


def train(env, cfg):
    # Extract training parameters
    K = cfg["K"]
    H = cfg["H"]
    delta_coeff = cfg["delta_coeff"]
    conf_coeff = cfg["conf_coeff"]

    # Initialize utility variables
    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    num_actions = env.R_list[0].shape[1]
    ref_arm, ref_state = 1, 2  # TODO
    # Initialze placeholders
    W = np.zeros((num_arms, num_states, num_arms, num_states))
    P_hat = np.ones((num_arms, num_states, num_states, num_actions)) / num_states
    Z_sa = np.zeros((num_arms, num_states, num_actions))
    Z_sas = np.zeros((num_arms, num_states, num_states, num_actions))

    F_tilde = np.ones((num_arms, num_states, num_arms, num_states)) * 0.5
    F_hat = np.ones((num_arms, num_states, num_arms, num_states)) * 0.5
    delta = np.ones((num_arms, num_states, num_actions)) * delta_coeff

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

    # Start traing
    for k in tqdm(range(K)):
        # check_stochasticity(P_hat)
        # Compute the corresponding index policy
        Q_n_s = np.log((1 - F_tilde[ref_arm][ref_state]) / F_tilde[ref_arm][ref_state])
        ##compute the policy
        W_sas = compute_ELP(
            delta, P_hat, env.arm_constraint, num_states, num_actions, Q_n_s, num_arms
        )
        W_sa = np.sum(W_sas, axis=2)
        index_matrix = W_sa[:, :, 1] / (W_sa[:, :, 0] + W_sa[:, :, 1])
        index_matrix = np.nan_to_num(index_matrix, nan=0.0)

        # Evaluate the policy
        train_curve.append(eval(env, index_matrix))
        rand_curve.append(eval(env, np.random.rand(num_arms, num_states)))
        opt_curve.append(eval(env, env.opt_index))

        # Compute recosntruction losses
        index_error.append(np.linalg.norm(index_matrix - env.opt_index))
        F_error.append(np.linalg.norm(F_tilde - F_true))
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
                    delta[arm_id, s, a] = (delta_coeff) * np.sqrt(
                        1 / 0.002 * (Z_sa[arm_id, s, a])
                    )  # TODO make closer to formula?
                    P_hat[arm_id, s, s_dash, a] = Z_sas[
                        arm_id, s, s_dash, a
                    ] / np.maximum(
                        1, Z_sa[arm_id, s, a]
                    )  # TODO
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
                    F_tilde[winner, s_list[winner], loser, s_list[loser]] = F_hat[
                        winner, s_list[winner], loser, s_list[loser]
                    ] + conf_coeff * np.sqrt(
                        1 / battle_count
                    )  # Make this closer to the actual formula?
                    F_tilde = np.clip(F_tilde, 1e-6, 1 - 1e-6)
                s_list = s_dash_list

        # Construct set of plausible transition kernels (i.e compute its center and radii)
        # P_hat = Z_sas / np.maximum(Z_sa[:, :, np.newaxis, :], 1)

    performance = {
        "train_curve": train_curve,
        "rand_curve": rand_curve,
        "opt_curve": opt_curve,
    }
    loss = {"index_error": index_error, "F_error": F_error, "P_error": P_error}
    print(
        "Training complete\n",
        f"index_learnt:\n {index_matrix}\n",
        f"true_index:\n {env.opt_index}",
    )
    return performance, loss


# Evaluate the policy
def eval(env, index_matrix):
    s_list = env.reset()
    total_reward = 0
    for t in range(env.T):
        action = apply_index_policy(s_list, index_matrix, env.arm_constraint)
        s_dash_list, reward, _, _, _ = env.step(action)
        total_reward += reward
        s_list = s_dash_list
    return total_reward
