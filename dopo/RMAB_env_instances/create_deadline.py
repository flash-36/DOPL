# import numpy as np


# def generate_deadline_matrices(max_charge, max_deadline):
#     """
#     Generate a random transition (P) and reward (R) matrix.
#     """
#     # Initialize the transition matrix P (shape: num_states x num_states x num_actions)
#     num_actions = 2
#     possible_charges = np.arange(max_charge + 1)
#     possible_deadlines = np.arange(max_deadline + 1)
#     possible_states = []
#     for charge in possible_charges:
#         for deadline in possible_deadlines:
#             if deadline > 0:
#                 possible_states.append((charge, deadline))
#     # Remove the state (0,12) as it is not reachable
#     # possible_states.remove((0, 12))
#     possible_states.append((0, 0))  # Empty slot
#     num_states = len(possible_states)

#     P = np.zeros((num_states, num_states, num_actions))
#     R = np.zeros((num_states, num_actions))

#     # Compute R
#     c = 0.5  # Charging cost
#     for state in possible_states:
#         for action in range(num_actions):
#             charge, deadline = state
#             if charge > 0 and deadline > 1:
#                 R[possible_states.index(state), action] = (1 - c) * action
#             elif charge > 0 and deadline == 1:
#                 R[possible_states.index(state), action] = (1 - c) * action - 0.2 * (
#                     charge - action
#                 ) ** 2
#             else:
#                 R[possible_states.index(state), action] = 0
#     # Normalize R to be between 0 and 1
#     R = (R - np.min(R)) / (np.max(R) - np.min(R))

#     # Compute P
#     for action in range(num_actions):
#         for state in possible_states:
#             charge, deadline = state
#             if deadline > 1:
#                 next_state = (max(charge - action, 0), deadline - 1)
#                 P[
#                     possible_states.index(state),
#                     possible_states.index(next_state),
#                     action,
#                 ] = 1
#             else:
#                 P[possible_states.index(state), :, action] = (
#                     0.7 * np.ones(num_states) / (num_states - 1)
#                 )
#                 P[
#                     possible_states.index(state), possible_states.index((0, 0)), action
#                 ] = 0.3
#                 # for new_deadline in range(2, max_deadline + 1):
#                 #     P[
#                 #         possible_states.index(state),
#                 #         possible_states.index((0, new_deadline)),
#                 #         action,
#                 #     ] = 0.0
#     # Check if P is stochastic
#     prob_check = np.allclose(P.sum(axis=1), 1, atol=1e-4)  # Allow some tolerance
#     print(P, R)
#     print("Probability Check Passed:", prob_check)
#     print(f"num_states: {num_states}")
#     return P, R


# def generate_and_save_multiple_mdps(num_mdps, max_charge, max_deadline):
#     """
#     Generate and save n MDPs with random P and R matrices.
#     Each file will be saved with a unique name.
#     """
#     for i in range(1, num_mdps + 1):
#         P, R = generate_deadline_matrices(max_charge, max_deadline)

#         # Save the transition matrix P to a .npy file
#         np.save(f"original_deadline_arm_type_{i}_transitions.npy", P)

#         # Save the reward matrix R to a .npy file
#         np.save(f"original_deadline_arm_type_{i}_rewards.npy", R)
#         print(
#             f"Saved MDP {i}: 'original_deadline_arm_type_{i}_transitions.npy' and 'original_deadline_arm_type_{i}_rewards.npy'"
#         )


# # Example usage
# def main():
#     num_mdps = 1
#     max_charge = 9
#     max_deadline = 12

#     generate_and_save_multiple_mdps(num_mdps, max_charge, max_deadline)


# if __name__ == "__main__":
#     main()

# action independant reward version of deadline environment
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
        np.save(f"deadline_{max_deadline}x{max_charge}_arm_type_{i}_transitions.npy", P)

        # Save the reward matrix R to a .npy file
        np.save(f"deadline_{max_deadline}x{max_charge}_arm_type_{i}_rewards.npy", R)
        print(
            f"Saved MDP {i}: 'deadline_{max_deadline}x{max_charge}_arm_type_{i}_transitions.npy' and 'deadline_{max_deadline}x{max_charge}_arm_type_{i}_rewards.npy'"
        )


# Example usage
def main():
    num_mdps = 1
    max_charge = 6
    max_deadline = 6

    generate_and_save_multiple_mdps(num_mdps, max_charge, max_deadline)


if __name__ == "__main__":
    main()
