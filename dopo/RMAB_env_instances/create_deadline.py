import numpy as np


def generate_deadline_matrices(max_charge, max_deadline):
    """
    Generate a random transition (P) and reward (R) matrix.
    """
    # Initialize the transition matrix P (shape: num_states x num_states x num_actions)
    num_actions = 2
    possible_charges = np.arange(max_charge + 1)
    possible_deadlines = np.arange(max_deadline + 1)
    possible_states = []
    for charge in possible_charges:
        for deadline in possible_deadlines:
            possible_states.append((charge, deadline))
    num_states = len(possible_states)

    P = np.zeros((num_states, num_states, num_actions))
    R = np.zeros(num_states)

    # Compute R
    for state in possible_states:
        charge, deadline = state
        if charge == 0:
            R[possible_states.index(state)] = max_charge
        elif deadline == 0:
            R[possible_states.index(state)] = -charge
        else:
            R[possible_states.index(state)] = 0
    # Reward is same for both actions
    R = np.tile(R[:, np.newaxis], (1, num_actions))
    # Normalize R to be between 0 and 1
    R = (R - np.min(R)) / (np.max(R) - np.min(R))

    # Compute P
    for action in range(num_actions):
        for state in possible_states:
            charge, deadline = state
            if charge != 0 and deadline != 0:
                next_state = (charge - action, deadline - 1)
                P[
                    possible_states.index(state),
                    possible_states.index(next_state),
                    action,
                ] = 1
            else:
                P[possible_states.index(state), :, action] = (
                    np.ones(num_states) / num_states
                )
    # Check if P is stochastic
    prob_check = np.allclose(P.sum(axis=1), 1, atol=1e-4)  # Allow some tolerance
    print(P, R)
    print("Probability Check Passed:", prob_check)
    print(f"num_states: {num_states}")
    return P, R


def generate_and_save_multiple_mdps(num_mdps, max_charge, max_deadline):
    """
    Generate and save n MDPs with random P and R matrices.
    Each file will be saved with a unique name.
    """
    for i in range(1, num_mdps + 1):
        P, R = generate_deadline_matrices(max_charge, max_deadline)

        # Save the transition matrix P to a .npy file
        np.save(f"deadline_arm_type_{i}_transitions.npy", P)

        # Save the reward matrix R to a .npy file
        np.save(f"deadline_arm_type_{i}_rewards.npy", R)
        print(
            f"Saved MDP {i}: 'deadline_arm_type_{i}_transitions.npy' and 'deadline_arm_type_{i}_rewards.npy'"
        )


# Example usage
def main():
    num_mdps = 1
    max_charge = 5
    max_deadline = 5

    generate_and_save_multiple_mdps(num_mdps, max_charge, max_deadline)


if __name__ == "__main__":
    main()
