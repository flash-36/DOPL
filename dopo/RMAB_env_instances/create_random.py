import numpy as np


def generate_random_matrices(num_states, num_actions):
    """
    Generate a random transition (P) and reward (R) matrix.
    """
    # Initialize the transition matrix P (shape: num_states x num_states x num_actions)
    P = np.zeros((num_states, num_states, num_actions))

    # Fill P with valid stochastic rows
    for action in range(num_actions):
        for state in range(num_states):
            P[state, :, action] = np.random.dirichlet(np.ones(num_states))

    # Initialize the reward matrix R (shape: num_states x num_actions)
    single_action_rewards = np.random.uniform(0, 100, num_states)
    R = np.tile(single_action_rewards[:, np.newaxis], (1, num_actions))

    return P, R


def generate_and_save_multiple_mdps(num_mdps, num_states, num_actions):
    """
    Generate and save n MDPs with random P and R matrices.
    Each file will be saved with a unique name.
    """
    for i in range(1, num_mdps + 1):
        P, R = generate_random_matrices(num_states, num_actions)

        # Save the transition matrix P to a .npy file
        np.save(f"random_arm_type_{i}_transitions.npy", P)

        # Save the reward matrix R to a .npy file
        np.save(f"random_arm_type_{i}_rewards.npy", R)

        print(
            f"Saved MDP {i}: 'random_arm_type_{i}_transitions.npy' and 'random_arm_type_{i}_rewards.npy'"
        )


# Example usage
def main():
    num_mdps = 5
    num_states = 10
    num_actions = 2

    generate_and_save_multiple_mdps(num_mdps, num_states, num_actions)


if __name__ == "__main__":
    main()
