import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .restless_arm import RestlessArmEnv


class MultiArmRestlessEnv(gym.Env):
    """
    Environment for a multi-arm restless bandit.
    The environment consists of multiple restless arms, each of which is a Markov Decision Process.
    Info dict contains the rewards obtained from each arm.
    """

    def __init__(self, arm_constraint, P_list, R_list, initial_dist=None):
        super(MultiArmRestlessEnv, self).__init__()
        assert len(P_list) == len(
            R_list
        ), "Number of transition matrices and reward matrices must match"
        assert all(
            P.shape[0] == P.shape[1] and P.shape[0] == R.shape[0]
            for P, R in zip(P_list, R_list)
        ), "Number of states in the Ps and Rs must match"
        self.arm_constraint = arm_constraint
        self.arms = [RestlessArmEnv(P, R, initial_dist) for P, R in zip(P_list, R_list)]
        self.action_space = spaces.MultiDiscrete([R.shape[1] for R in R_list])
        self.observation_space = spaces.MultiDiscrete([R.shape[0] for R in R_list])

    def step(self, action):
        assert sum(action) <= self.arm_constraint, "Action constraint on arms pulled"
        rewards = []
        states = []
        for i, arm in enumerate(self.arms):
            state, reward, _, _, _ = arm.step(action[i])
            rewards.append(reward)
            states.append(state)
        assert self.observation_space.contains(states)
        ter = tru = False  # No terminal state in bandit environments
        return states, sum(rewards), ter, tru, {"arm_rewards": rewards}

    def reset(self):
        return [arm.reset() for arm in self.arms]

    def render(self, mode="human"):
        for i, arm in enumerate(self.arms):
            arm.render()
