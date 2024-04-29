import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.restless_arm import RestlessArmEnv


class MultiArmRestlessEnv(gym.Env):
    """Environment for a multi-arm restless bandit problem."""

    def __init__(self, num_arms, arm_constraint, P, R, initial_dist=None):
        super(MultiArmRestlessEnv, self).__init__()
        self.num_arms = num_arms
        self.arm_constraint = arm_constraint
        self.arms = [RestlessArmEnv(P, R, initial_dist) for _ in range(num_arms)]
        num_states_per_arm, num_actions_per_arm = R.shape
        self.action_space = spaces.MultiDiscrete([num_actions_per_arm] * num_arms)
        self.observation_space = spaces.Tuple(
            [spaces.Discrete(num_states_per_arm) for _ in range(num_arms)]
        )

    def step(self, action):
        assert sum(action) <= self.arm_constraint, "Action constraint on arms pulled"
        rewards = []
        states = []
        for i, arm in enumerate(self.arms):
            state, reward, _, _ = arm.step(action[i])
            rewards.append(reward)
            states.append(state)
        assert self.observation_space.contains(states)
        ter = tru = False  # No terminal state in bandit environments
        return states, sum(rewards), ter, tru, {}

    def reset(self):
        return [arm.reset() for arm in self.arms]

    def render(self, mode="human"):
        for i, arm in enumerate(self.arms):
            arm.render()
