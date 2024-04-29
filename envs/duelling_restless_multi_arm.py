from envs import MultiArmRestlessEnv


class MultiArmRestlessDuellingEnv(MultiArmRestlessEnv):
    """Environment for a duelling multi-arm restless bandit problem."""

    def __init__(self, num_arms, arm_constraint, P, R, initial_dist=None):
        super().__init__(num_arms, arm_constraint, P, R, initial_dist)

    def step(self, action):
        assert sum(action) <= self.arm_constraint, "Exactly K arms must be pulled"

        # Custom behavior can be added here
        print(f"Action taken: {action}")

        # Call the original step method or redefine it
        states, reward, ter, tru, info = super().step(action)

        # Modify the behavior post original step execution
        # Example: add custom logging, modify the rewards based on some condition, etc.
        # For demonstration, let's add a custom log message:
        print(f"Total reward obtained: {reward}")

        # Optionally modify the outputs
        return states, reward, ter, tru, info
