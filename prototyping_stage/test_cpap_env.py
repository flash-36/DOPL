import numpy as np
from dopo.envs import MultiArmRestlessDuellingEnv

from dopo.utils import load_arm


def main():
    # Load the transition and reward matrices for both arm types
    P1, R1 = load_arm("cpap_arm_type_1")
    P2, R2 = load_arm("cpap_arm_type_2")

    # Initialize environment with two arms of each type
    P_list = [P1, P1, P2, P2]
    R_list = [R1, R1, R2, R2]
    arm_constraint = 3  # Define how many arms can be chosen at once

    # Create an instance of the duelling environment
    env = MultiArmRestlessDuellingEnv(arm_constraint, P_list, R_list)

    # Reset the environment to start
    states = env.reset()
    while True:
        print("States:", states)

        # Take a step using random actions within the constraints
        actions = np.array(
            [np.random.randint(0, 2) for _ in range(len(P_list))], dtype=int
        )  # Random actions for each arm
        if actions.sum() > arm_constraint:
            continue
        print("Actions taken:", actions)
        next_states, reward, _, _, info = env.step(actions)

        # Output the results of the step
        print("Next states:", next_states)
        print("Reward received:", reward)
        print("Detailed info:", info)
        states = next_states

        a = input()


if __name__ == "__main__":
    main()
