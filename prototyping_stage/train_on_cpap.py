import numpy as np
from dopo.envs import MultiArmRestlessDuellingEnv

from dopo.utils import load_arm, ELP

H = 20
K = 100
delta_coeff = 0.1
conf_coeff = 0.1


def main():
    # Load the transition and reward matrices for both arm types
    P1, R1 = load_arm("cpap_arm_type_1")
    P2, R2 = load_arm("cpap_arm_type_2")

    # Initialize environment with two arms of each type
    P_list = [P1] * 5 + [P2] * 5
    R_list = [R1] * 5 + [R2] * 5
    arm_constraint = 4  # Define how many arms can be chosen at once

    # Create an instance of the duelling environment
    env = MultiArmRestlessDuellingEnv(arm_constraint, P_list, R_list)

    train(env)


def train(env):
    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    num_actions = env.R_list[0].shape[1]
    W = np.zeros((num_arms, num_states, num_arms, num_states))
    C = np.zeros((num_arms, num_states, num_arms, num_states))
    P_hat = np.ones((num_arms, num_states, num_states, num_actions)) / num_states
    Z_sa = np.zeros((num_arms, num_states, num_actions))
    Z_sas = np.zeros((num_arms, num_states, num_actions, num_states))
    for k in range(K):
        s_list = env.reset()
        for h in range(H):
            # action = policy(s_list)
            action = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
            s_list, reward, _, _, info = env.step(action)
            for record in info["duelling_results"]:
                winner, loser = record
                W[winner, s_list[winner], loser, s_list[loser]] += 1
                count = (
                    W[winner, s_list[winner], loser, s_list[loser]]
                    + W[loser, s_list[loser], winner, s_list[winner]]
                )


if __name__ == "__main__":
    main()
