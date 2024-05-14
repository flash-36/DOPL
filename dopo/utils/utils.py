from pathlib import Path
import os
import numpy as np
import math
import pulp as p
import sys

RMAB_PATH = os.path.join(Path(__file__).parent.parent, "RMAB_env_instances")


def check_stochasticity(P):
    for n in range(P.shape[0]):
        for s in range(P.shape[1]):
            for a in range(P.shape[3]):
                if not math.isclose(np.sum(P[n, s, :, a]), 1, abs_tol=1e-2):
                    print(
                        f"Arm {n}, State {s}, Action {a} is not stochastic summing to {np.sum(P[n, s, :, a])}"
                    )
                    return False
    return True


def load_arm(arm_name):
    P = np.load(os.path.join(RMAB_PATH, f"{arm_name}_transitions.npy"))
    R = np.load(os.path.join(RMAB_PATH, f"{arm_name}_rewards.npy"))
    return P, R


def compute_ELP(delta, P_hat, budget, n_state, n_action, Reward, n_arms):
    """
    delta: delta_n(s,a) : n_arms*n_state*n_action
    P_hat: P_hat_n(s'/s,a): n_arms* n_state*n_state*n_action
    Reward: R_n(s): n_arms* n_state
    budget: SCALAR
    """

    index_policy = np.zeros((n_arms, n_state, n_state, n_action))
    opt_prob = p.LpProblem("ExtendedLP", p.LpMaximize)
    idx_p_keys = [
        (n, s, a, s_dash)
        for n in range(n_arms)
        for s in range(n_state)
        for a in range(n_action)
        for s_dash in range(n_state)
    ]
    w = p.LpVariable.dicts("w", idx_p_keys, lowBound=0, upBound=1, cat="continous")

    r = Reward.copy()

    # objective equation
    opt_prob += p.lpSum(
        [
            w[(n, s, a, s_dash)] * r[n][s]
            for n in range(n_arms)
            for s in range(n_state)
            for a in range(n_action)
            for s_dash in range(n_state)
        ]
    )
    # list1 = [r[n][s] for n in range(n_arms) for s in range(n_state)]*self.

    # Budget Constraint

    opt_prob += (
        p.lpSum(
            [
                w[(n, s, a, s_dash)] * a
                for n in range(n_arms)
                for s in range(n_state)
                for a in range(n_action)
                for s_dash in range(n_state)
            ]
        )
        - budget
        <= 0
    )

    for n in range(n_arms):
        for s in range(n_state):
            w_list = [
                w[(n, s, a, s_dash)]
                for a in range(n_action)
                for s_dash in range(n_state)
            ]
            w_1_list = [
                w[(n, s_dash, a_dash, s)]
                for a_dash in range(n_action)
                for s_dash in range(n_state)
            ]
            opt_prob += p.lpSum(w_list) - p.lpSum(w_1_list) == 0

    for n in range(n_arms):

        a_list = [
            w[(n, s, a, s_dash)]
            for s in range(n_state)
            for a in range(n_action)
            for s_dash in range(n_state)
        ]
        opt_prob += p.lpSum(a_list) - 1 == 0

    # Extended part of the Linear Programming
    for n in range(n_arms):
        for s in range(n_state):
            for a in range(n_action):
                for s_dash in range(n_state):

                    b_list = [w[(n, s, a, s_dash)] for s_dash in range(n_state)]
                    opt_prob += (
                        w[(n, s, a, s_dash)]
                        - (P_hat[n][s][s_dash][a] + delta[n][s][a]) * p.lpSum(b_list)
                        <= 0
                    )

                    opt_prob += (
                        -1 * w[(n, s, a, s_dash)]
                        + p.lpSum(b_list) * (P_hat[n][s][s_dash][a] - delta[n][s][a])
                        <= 0
                    )

    status = opt_prob.solve(p.PULP_CBC_CMD(msg=0))
    # status = opt_prob.solve(p.GUROBI(msg=0))
    # status = opt_prob.solve(p.GLPK_CMD(msg=0))
    if p.LpStatus[status] != "Optimal":
        return None
    # assert p.LpStatus[status] == "Optimal", "No feasible solution :("

    for n in range(n_arms):
        for s in range(n_state):
            for a in range(n_action):
                for s_dash in range(n_state):
                    index_policy[n, s, s_dash, a] = w[(n, s, a, s_dash)].varValue

                    if (index_policy[n, s, s_dash, a]) < 0 and index_policy[
                        n, s, s_dash, a
                    ] > -0.001:
                        index_policy[n, s, s_dash, a] = 0
                    elif index_policy[n, s, s_dash, a] < -0.001:
                        print("Invalid Value")
                        sys.exit()

    return index_policy


def compute_optimal(Reward, budget, P, n_arms, n_state, n_action):

    optimal_policy = np.zeros((n_arms, n_state, n_action))
    opt_prob = p.LpProblem("OptimalLP", p.LpMaximize)
    p_keys = [
        (n, s, a)
        for n in range(n_arms)
        for s in range(n_state)
        for a in range(n_action)
    ]
    w = p.LpVariable.dicts("w", p_keys, lowBound=0, upBound=1, cat="continous")

    r = {}
    for n in range(n_arms):
        r[n] = {}

        for state in range(n_state):
            r[n][state] = Reward[n][state]
    # objective function
    opt_prob += p.lpSum(
        [
            w[(n, s, a)] * r[n][s]
            for n in range(n_arms)
            for s in range(n_state)
            for a in range(n_action)
        ]
    )
    # Budget Constraint
    opt_prob += (
        p.lpSum(
            [
                w[(n, s, a)] * a
                for n in range(n_arms)
                for s in range(n_state)
                for a in range(n_action)
            ]
        )
        - budget
        <= 0
    )

    for n in range(n_arms):
        w_list = [w[(n, s, a)] for s in range(n_state) for a in range(n_action)]
        opt_prob += p.lpSum(w_list) - 1 == 0

    for n in range(n_arms):
        for s in range(n_state):
            for a in range(n_action):
                opt_prob += w[(n, s, a)] >= 0

    for n in range(n_arms):
        for s in range(n_state):
            a_list = [w[(n, s, a)] for a in range(n_action)]
            b_list = [
                w[(n, s_dash, a_dash)] * P[n][s_dash][s][a_dash]
                for s_dash in range(n_state)
                for a_dash in range(n_action)
            ]
            opt_prob += p.lpSum(a_list) - p.lpSum(b_list) == 0

    status = opt_prob.solve(p.PULP_CBC_CMD(msg=0))
    assert p.LpStatus[status] == "Optimal", "No feasible solution :("

    for n in range(n_arms):
        for s in range(n_state):
            for a in range(n_action):
                optimal_policy[n, s, a] = w[(n, s, a)].varValue
                if (optimal_policy[n, s, a]) < 0 and optimal_policy[n, s, a] > -0.001:
                    optimal_policy[n, s, a] = 0
                elif optimal_policy[n, s, a] < -0.001:
                    print("Invalid Value")
                    sys.exit()

    opt_value = p.value(opt_prob.objective)
    return opt_value, optimal_policy
