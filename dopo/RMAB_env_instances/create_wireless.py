import numpy as np


def generate_wireless_matrices(max_data, num_channels):
    """
    Generate transition (P) and reward (R) matrix.
    """
    # Initialize the transition matrix P (shape: num_states x num_states x num_actions)
    num_actions = 2
    possible_data_levels = np.arange(max_data + 1)
    possible_channel_qualities = np.arange(num_channels)
    possible_states = []
    for data_level in possible_data_levels:
        for channel_quality in possible_channel_qualities:
            possible_states.append((data_level, channel_quality))
    num_states = len(possible_states)

    P = np.zeros((num_states, num_states, num_actions))
    R = np.zeros(num_states)

    # Compute R
    for state in possible_states:
        data_level, channel_quality = state
        R[possible_states.index(state)] = -data_level
        if data_level == 0:
            R[possible_states.index(state)] = 4 * max_data
    # Reward is same for both actions
    R = np.tile(R[:, np.newaxis], (1, num_actions))
    # Normalize R to be between 0 and 1
    R = (R - np.min(R)) / (np.max(R) - np.min(R))

    # Compute P
    for action in range(num_actions):
        for state in possible_states:
            data_level, channel_quality = state
            if data_level != 0:
                possible_next_states = [
                    (
                        max(0, data_level - channel_quality * action),
                        next_channel_quality,
                    )
                    for next_channel_quality in possible_channel_qualities
                ]
                for next_state in possible_next_states:
                    P[
                        possible_states.index(state),
                        possible_states.index(next_state),
                        action,
                    ] = 1 / len(possible_next_states)
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


def generate_and_save_multiple_mdps(num_mdps, max_data, num_channels):
    """
    Generate and save n MDPs with random P and R matrices.
    Each file will be saved with a unique name.
    """
    for i in range(1, num_mdps + 1):
        P, R = generate_wireless_matrices(max_data, num_channels)

        # Save the transition matrix P to a .npy file
        np.save(f"wireless_arm_type_{i}_transitions.npy", P)

        # Save the reward matrix R to a .npy file
        np.save(f"wireless_arm_type_{i}_rewards.npy", R)
        print(
            f"Saved MDP {i}: 'wireless_arm_type_{i}_transitions.npy' and 'wireless_arm_type_{i}_rewards.npy'"
        )


# Example usage
def main():
    num_mdps = 1
    max_data = 6
    num_channels = 3

    generate_and_save_multiple_mdps(num_mdps, max_data, num_channels)


if __name__ == "__main__":
    main()
