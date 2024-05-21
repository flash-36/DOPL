import numpy as np


def main():
   
    P_arm1, R_arm1 = satellite60()
    verify_transitions(P_arm1, "Arm 1")
    save_matrices(P_arm1, R_arm1, "arm_type_1")

    P_arm2, R_arm2 = satellite40()
    verify_transitions(P_arm2, "Arm 2")
    save_matrices(P_arm2, R_arm2, "arm_type_2")

    P_arm3, R_arm3 = satellite70()
    verify_transitions(P_arm3, "Arm 3")
    save_matrices(P_arm3, R_arm3, "arm_type_3")

    P_arm4, R_arm4 = satellite80()
    verify_transitions(P_arm4, "Arm 4")
    save_matrices(P_arm4, R_arm4, "arm_type_4")


def satellite40():
    n_states = 2
    n_actions = 2
    P = np.zeros((n_states, n_states, n_actions))
    P[:, :, 0] = [
        [0.0845, 0.9155],
        [0.0811, 0.9189],
    ]  # Action 0
    P[:, :, 1] = [
        [0.9155, 0.0845],
        [0.9189, 0.0811],
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

def satellite80():
    n_states = 2
    n_actions = 2
    P = np.zeros((n_states, n_states, n_actions))
    P[:, :, 0] = [
        [0.0732, 0.9268],
        [0.2667, 0.7333],
    ]  # Action 0
    P[:, :, 1] = [
        [0.9268, 0.0732],
        [0.7333, 0.2667],
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

def satellite70():
    n_states = 2
    n_actions = 2
    P = np.zeros((n_states, n_states, n_actions))
    P[:, :, 0] = [
        [0.0845, 0.9155],
        [0.2069, 0.7931],
    ]  # Action 0
    P[:, :, 1] = [
        [0.9155, 0.0845],
        [0.7931, 0.2069],
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


def satellite60():
    n_states = 2
    n_actions = 2
    P = np.zeros((n_states, n_states, n_actions))
    P[:, :, 0] = [
        [0.9043, 0.0957],
        [0.8, 0.2],
    ]  # Action 0
    P[:, :, 1] = [
        [0.0957, 0.9043],
        [0.2, 0.8],
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
    np.save(f"LMSS_{arm_type}_transitions.npy", P)
    np.save(f"LMSS_{arm_type}_rewards.npy", R)
    print(f"Matrices for {arm_type} saved successfully.")


if __name__ == "__main__":
    main()
