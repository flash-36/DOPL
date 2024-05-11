# videostreaming environment
import numpy as np


def main():
    # Define the number of states and actions
    n_states = 10
    n_actions = 2

    # Initialize and define transition and reward matrices for each arm separately
    P_arm1, R_arm1 = initialize_arm1(n_states, n_actions)

    # Verify the transition matrices
    verify_transitions(P_arm1, "Arm 1")

    # Save matrices to files
    save_matrices(P_arm1, R_arm1, "arm_type_1")


def initialize_arm1(n_states, n_actions):
    # Transition matrices for arm 1
    P = np.zeros((n_states, n_states, n_actions))
    P[:, :, 0] = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.6, 0.4, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0.6, 0.4, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.6, 0.4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.6, 0.4, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.6, 0.4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0.6, 0.4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.6, 0.4, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0.6, 0.4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.6, 0.4],
    ]  # Action 0

    P[:, :, 1] = [
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]  # Action 1

    # Reward matrix for arm 1
    R = np.zeros((n_states, n_actions))
    for state_index in range(n_states):
        if state_index == 0:
            R[state_index, :] = 0
        else:
            R[state_index, :] = 1 / (state_index + 1)
    print(P)
    print(R)
    return P, R


def verify_transitions(P, arm_description):
    row_sums = np.sum(P, axis=1)
    if not np.allclose(row_sums, np.ones((P.shape[0], P.shape[2])), atol=1e-8):
        print(f"{arm_description}: Row sums not 1 - ERROR")
        print("Row sums:", row_sums)
    else:
        print(f"{arm_description}: Rows sum to 1 - OK")


def save_matrices(P, R, arm_type):
    np.save(f"vs_{arm_type}_transitions.npy", P)
    np.save(f"vs_{arm_type}_rewards.npy", R)
    print(f"Matrices for {arm_type} saved successfully.")


if __name__ == "__main__":
    main()
