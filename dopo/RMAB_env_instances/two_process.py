import numpy as np


def main():
    # Define the number of states and actions

    # Initialize and define transition and reward matrices for each arm separately
    P_arm1, R_arm1 = process_a()

    # Verify the transition matrices
    verify_transitions(P_arm1, "Arm 1")

    # Save matrices to files
    save_matrices(P_arm1, R_arm1, "arm_type_1")
    P_arm2, R_arm2 = process_b()
    verify_transitions(P_arm2, "Arm 2")
    save_matrices(P_arm2, R_arm2, "arm_type_2")
    T=[None]*2
    # print(P_arm1.shape)
    T[0] = P_arm1.transpose(0,2,1)
    T[1] = P_arm2.transpose(0,2,1)
    print("The matrix you need for LPQL is",np.array(T))


def process_a():
    n_states = 2
    n_actions = 2
    P = np.zeros((n_states, n_states, n_actions))
    P[:, :, 0] = [
        [0.8, 0.2],
        [0.75, 0.25],
    ]  # Action 0
    P[:, :, 1] = [
        [0.7, 0.3],
        [0.01, 0.99],
    ]  # Action 1

    # Reward matrix for arm 1
    R = np.zeros((n_states, n_actions))
    for state_index in range(n_states):
        if state_index == 0:
            R[state_index, :] = 0
        else:
            R[state_index, :] = 1
    print(P)
    print(R)
    return P, R


def process_b():
    n_states = 2
    n_actions = 2
    P = np.zeros((n_states, n_states, n_actions))
    P[:, :, 0] = [
        [0.95, 0.05],
        [0.25, 0.75],
    ]  # Action 0
    P[:, :, 1] = [
        [0.01, 0.99],
        [0.01, 0.99],
    ]  # Action 0

    # Reward matrix for arm 1
    R = np.zeros((n_states, n_actions))
    for state_index in range(n_states):
        if state_index == 0:
            R[state_index, :] = 0
        else:
            R[state_index, :] = 1
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
    np.save(f"two_process_{arm_type}_transitions.npy", P)
    np.save(f"two_process_{arm_type}_rewards.npy", R)
    print(f"Matrices for {arm_type} saved successfully.")


if __name__ == "__main__":
    main()
