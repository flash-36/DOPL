from pathlib import Path
import os
import numpy as np
import numpy as np
import math
import time
import pulp as p
import sys

RMAB_PATH = os.path.join(Path(__file__).parent.parent, "RMAB_env_instances")


def load_arm(arm_name):
    P = np.load(os.path.join(RMAB_PATH, f"{arm_name}_transitions.npy"))
    R = np.load(os.path.join(RMAB_PATH, f"{arm_name}_rewards.npy"))
    return P, R


class ELP:
    def __init__(self, delta, P_hat, budget, n_state, n_actions, Reward, n_arms):
        """
        delta: delta_n(s,a) : n_arms*n_state*n_action
        P_hat: P_hat_n(s'/s,a): n_arms* n_state*n_state*n_action
        Reward: R_n(s): n_arms* n_state
        budget: SCALAR
        """
        self.delta = delta
        self.P_hat = P_hat
        self.budget = budget
        self.n_state = n_state
        self.n_action = n_actions
        self.R = Reward
        self.n_arms = n_arms

    def compute_ELP(self):

        
        index_policy = np.zeros(
            (self.n_arms, self.n_state, self.n_state, self.n_action)
        )
        opt_prob = p.LpProblem("ExtendedLP", p.LpMaximize)
        idx_p_keys = [
            (n, s, a, s_dash)
            for n in range(self.n_arms)
            for s in range(self.n_state)
            for a in range(self.n_action)
            for s_dash in range(self.n_state)
        ]
        w = p.LpVariable.dicts("w", idx_p_keys, lowBound=0, upBound=1, cat="continous")

        r = self.R.copy()

        # objective equation
        opt_prob += p.lpSum(
            [
                w[(n, s, a, s_dash)] * r[n][s]
                for n in range(self.n_arms)
                for s in range(self.n_state)
                for a in range(self.n_action)
                for s_dash in range(self.n_state)
            ]
        )
        # list1 = [r[n][s] for n in range(self.n_arms) for s in range(self.n_state)]*self.

        # Budget Constraint
        

        opt_prob += (
            p.lpSum(
                [
                    w[(n, s, a, s_dash)] * a
                    for n in range(self.n_arms)
                    for s in range(self.n_state)
                    for a in range(self.n_action)
                    for s_dash in range(self.n_state)
                ]
            )
            - self.budget
            <= 0
        )



        for n in range(self.n_arms):
            for s in range(self.n_state):
                w_list = [
                    w[(n, s, a, s_dash)]
                    for a in range(self.n_action)
                    for s_dash in range(self.n_state)
                ]
                w_1_list = [
                    w[(n, s_dash, a_dash, s)]
                    for a_dash in range(self.n_action)
                    for s_dash in range(self.n_state)
                ]
                opt_prob += p.lpSum(w_list) - p.lpSum(w_1_list) == 0

        for n in range(self.n_arms):

            a_list = [
                w[(n, s, a, s_dash)]
                for s in range(self.n_state)
                for a in range(self.n_action)
                for s_dash in range(self.n_state)
            ]
            opt_prob += p.lpSum(a_list) -1 == 0


        # Extended part of the Linear Programming
        for n in range(self.n_arms):
            for s in range(self.n_state):
                for a in range(self.n_action):
                    for s_dash in range(self.n_state):

                        b_list = [
                            w[(n, s, a, s_dash)] for s_dash in range(self.n_state)
                        ]
                        opt_prob += (
                            w[(n, s, a, s_dash)]
                            - (self.P_hat[n][s_dash][s][a] + self.delta[n][s][a])
                            * p.lpSum(b_list)
                            <= 0
                        )

                        opt_prob += (
                            -1 * w[(n, s, a, s_dash)]
                            + p.lpSum(b_list)
                            * (self.P_hat[n][s_dash][s][a] - self.delta[n][s][a])
                            <= 0
                        )
        
        
        status = opt_prob.solve(p.PULP_CBC_CMD(msg=0))

        
        
        if p.LpStatus[status] != "Optimal":
            print("The policy is not optimal")
            return index_policy
        
        

        for n in range(self.n_arms):
            for s in range(self.n_state):
                for a in range(self.n_action):
                    for s_dash in range(self.n_state):
                        index_policy[n, s, s_dash, a] = w[(n, s, a, s_dash)].varValue
                        

                        if (index_policy[n, s, s_dash, a]) < 0 and index_policy[
                            n, s, s_dash, a
                        ] > -0.001:
                            index_policy[n, s, s_dash, a] = 0
                        elif index_policy[n, s, s_dash, a] < -0.001:
                            print("Invalid Value")
                            sys.exit()

        

        return index_policy


class Optimal:
    def __init__(self, reward, budget, P, n_arms, n_state, n_action):
        """
        P : n*s*s*a
        R: n*s
        w : n*s*a
        """
        self.R = reward
        self.P = P
        self.B = budget
        self.n_arms = n_arms
        self.n_state = n_state
        self.n_action = n_action

    def compute_optimal(self):

        optimal_policy = np.zeros((self.n_arms, self.n_state, self.n_action))
        prob = p.LpProblem("OptimalLP", p.LpMaximize)
        p_keys = [
            (n, s, a)
            for n in range(self.n_arms)
            for s in range(self.n_state)
            for a in range(self.n_action)
        ]
        w = p.LpVariable.dicts("w", p_keys, lowBound=0, upBound=1, cat="continous")

        r = {}
        for n in range(self.n_arms):
            r[n] = {}

            for state in range(self.n_state):
                r[n][state] = self.R[n][state]
        # objective function
        opt_prob += p.lpSum(
            [
                w[(n, s, a)] * r[n][s]
                for n in range(self.n_arms)
                for s in range(self.n_state)
                for a in range(self.n_action)
            ]
        )
        # Budget Constraint
        opt_prob += (
            p.lpSum(
                [
                    w[(n, s, a)] * a
                    for n in range(self.n_arms)
                    for s in range(self.n_state)
                    for a in range(self.n_action)
                ]
            )
            - self.budget
            <= 0
        )

        for n in range(self.arms):
            w_list = [
                w[(n, s, a)] for s in range(self.state) for a in range(self.n_action)
            ]
            opt_prob += p.lpSum(w_list) - 1 == 0

        for n in range(self.arms):
            for s in range(self.n_state):
                for a in range(self.actions):
                    opt_prob += w[(n, s, a)] >= 0

        for n in range(self.arms):
            for s in range(self.n_state):
                a_list = [w[(n, s, a)] for a in range(self.n_action)]
                b_list = [
                    w[(n, s_dash, a_dash)] * self.P[s_dash][a_dash][s]
                    for s_dash in range(self.n_state)
                    for a_dash in range(self.n_action)
                ]
                opt_prob += p.lpSum(a_list) - p.lpSum(b_list) == 0

        status = opt_prob.solve(p.PULP_CBC_CMD(gapRel=0.01, msg=0))
        if p.LpStatus[status] != "optimal":
            print("Optimal policy calculation failed")
            return optimal_policy

        for n in range(self.n_arms):
            for s in range(self.n_state):
                for a in range(self.n_action):
                    optimal_policy[n, s, a] = w[(n, s, a)].varValue
                    if (optimal_policy[n, s, a]) < 0 and optimal_policy[
                        n, s, a
                    ] > -0.001:
                        optimal_policy[n, s, a] = 0
                    elif optimal_policy[n, s, a] < -0.001:
                        print("Invalid Value")
                        sys.exit()

        return optimal_policy
