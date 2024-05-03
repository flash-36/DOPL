import numpy as np
from dopo.envs import MultiArmRestlessDuellingEnv

from dopo.utils import load_arm, ELP

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
    P_list = [P1] * 5 + [P2] * 5
    R_list = [R1] * 5 + [R2] * 5
    arm_constraint = 4  # Define how many arms can be pulled at once

    # Create an instance of the duelling environment
    env = MultiArmRestlessDuellingEnv(arm_constraint, P_list, R_list)

    train(env)


def eval(env, policy):
    s_list = env.reset()
    total_reward = 0
    for t in range(T):
        action = policy(s_list)
        s_dash_list, reward, _, _, _ = env.step(action)
        total_reward += reward
    return total_reward


def train(env):
    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    num_actions = env.R_list[0].shape[1]
    W = np.zeros((num_arms, num_states, num_arms, num_states))
    C = np.zeros((num_arms, num_states, num_arms, num_states))
    P_hat = np.ones((num_arms, num_states, num_states, num_actions)) / num_states
    Z_sa = np.zeros((num_arms, num_states, num_actions))
    Z_sas = np.zeros((num_arms, num_states, num_actions, num_states))
    train_curve = []
    for k in range(K):
        # Compute the policy
        # policy = ELP(P_hat, ....)
        # Evaluate the policy
        # train_curve.append(eval(env, policy))
        for h in range(H):
            s_list = env.reset()
            for t in range(T):
                # action = policy(s_list)
                action = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
                s_dash_list, reward, _, _, info = env.step(action)
                for arm_id, s, a, s_dash in zip(
                    range(num_arms), s_list, action, s_dash_list
                ):
                    Z_sa[arm_id, s, a] += 1
                    Z_sas[arm_id, s, a, s_dash] += 1
                for record in info["duelling_results"]:
                    winner, loser = record
                    W[winner, s_list[winner], loser, s_list[loser]] += 1
        C = W + W.transpose(2, 3, 0, 1)  # Check if the indentation level is correct
        F_hat = W / C
        F_tilde = F_hat + conf_coeff * np.sqrt(
            np.log(k) / C
        )  # Make this closer to the actual formula?
        # Construct set of plausible transition kernels (i.e compute its center and radii)
        P_hat = Z_sas / max(Z_sa[:, :, :, None], 1)  # TODO: Verify this functionality
        delta = delta_coeff * np.sqrt(
            np.log(k) / Z_sa
        )  # Make this closer to the actual formula?


if __name__ == "__main__":
    main()
