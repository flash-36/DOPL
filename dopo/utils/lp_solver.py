import numpy as np
import sys
import numpy as np
import pulp as p
import sys

from pyomo.environ import *


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


def compute_ELP_pyomo(delta, P_hat, budget, n_state, n_action, Reward, n_arms):
    """
    Convert the Extended Linear Program to use Pyomo with Gurobi, ensuring unique constraint handling.
    """
    model = ConcreteModel()

    # Sets for indexing
    model.N = RangeSet(0, n_arms - 1)
    model.S = RangeSet(0, n_state - 1)
    model.A = RangeSet(0, n_action - 1)
    model.S_dash = RangeSet(0, n_state - 1)

    # Decision variables
    model.w = Var(model.N, model.S, model.A, model.S_dash, bounds=(0, 1))

    # Objective function: Maximizing the weighted reward
    model.obj = Objective(
        expr=sum(
            model.w[n, s, a, s_dash] * Reward[n][s]
            for n in model.N
            for s in model.S
            for a in model.A
            for s_dash in model.S_dash
        ),
        sense=maximize,
    )

    # Budget Constraint: Total cost should not exceed the budget
    model.budget_constraint = Constraint(
        expr=sum(
            model.w[n, s, a, s_dash] * a
            for n in model.N
            for s in model.S
            for a in model.A
            for s_dash in model.S_dash
        )
        <= budget
    )

    # Flow balance constraints for each arm and state
    def flow_balance_constraints(model, n, s):
        return (
            sum(model.w[n, s, a, s_dash] for a in model.A for s_dash in model.S_dash)
            - sum(
                model.w[n, s_dash, a_dash, s]
                for a_dash in model.A
                for s_dash in model.S_dash
            )
            == 0
        )

    model.flow_balance = Constraint(model.N, model.S, rule=flow_balance_constraints)

    # Probability constraints ensuring that the sum over actions and states is 1 for each arm
    def probability_constraints(model, n):
        return (
            sum(
                model.w[n, s, a, s_dash]
                for s in model.S
                for a in model.A
                for s_dash in model.S_dash
            )
            == 1
        )

    model.probability = Constraint(model.N, rule=probability_constraints)

    # Extended constraints for delta and P_hat, ensuring transition probabilities are feasible
    def extended_constraints(model, n, s, a, s_dash):
        # breakpoint()
        return (
            model.w[n, s, a, s_dash]
            - (P_hat[n][s][s_dash][a] + delta[n][s][a])
            * sum(model.w[n, s, a, y] for y in model.S_dash)
            <= 0
        )

    def extended_constraints_1(model, n, s, a, s_dash):
        return (
            -model.w[n, s, a, s_dash]
            + sum(model.w[n, s, a, y] for y in model.S_dash)
            * (P_hat[n][s][s_dash][a] - delta[n][s][a])
            <= 0
        )

    model.extended = Constraint(
        model.N, model.S, model.A, model.S_dash, rule=extended_constraints
    )
    model.extended = Constraint(
        model.N, model.S, model.A, model.S_dash, rule=extended_constraints_1
    )

    # Solve the model using Gurobi
    solver = SolverFactory("gurobi")
    # solver.options["LogToConsole"] = True
    result = solver.solve(model, tee=True)

    # Process and return the results
    if (
        result.solver.status == SolverStatus.ok
        and result.solver.termination_condition == TerminationCondition.optimal
    ):
        # print("Optimal solution found!")
        index_policy = np.zeros((n_arms, n_state, n_state, n_action))
        for n in model.N:
            for s in model.S:
                for a in model.A:
                    for s_dash in model.S_dash:
                        index_policy[n, s, s_dash, a] = model.w[n, s, a, s_dash].value
        return index_policy, value(model.obj)
    elif result.solver.termination_condition == TerminationCondition.infeasible:
        raise RuntimeError("No feasible solution found!")
    else:
        raise RuntimeError("No feasible solution found! Something went wrong :(")


def compute_optimal_pyomo(Reward, budget, P, n_arms, n_state, n_action):
    model = ConcreteModel()

    # Define index sets
    model.N = RangeSet(0, n_arms - 1)
    model.S = RangeSet(0, n_state - 1)
    model.A = RangeSet(0, n_action - 1)

    # Define the decision variables
    model.w = Var(model.N, model.S, model.A, bounds=(0, 1))

    # Objective function: Maximize the weighted reward
    model.objective = Objective(
        expr=sum(
            model.w[n, s, a] * Reward[n][s]
            for n in model.N
            for s in model.S
            for a in model.A
        ),
        sense=maximize,
    )

    # Budget constraint
    model.budget_constraint = Constraint(
        expr=sum(
            model.w[n, s, a] * a for n in model.N for s in model.S for a in model.A
        )
        <= budget
    )

    # Sum of probabilities equals 1 for each arm
    def sum_prob_rule(model, n):
        return sum(model.w[n, s, a] for s in model.S for a in model.A) == 1

    model.sum_prob = Constraint(model.N, rule=sum_prob_rule)

    # Ensure probabilities are valid and transition probabilities are conserved
    def transition_rule(model, n, s):
        return sum(model.w[n, s, a] for a in model.A) == sum(
            model.w[n, s_dash, a_dash] * P[n][s_dash][s][a_dash]
            for s_dash in model.S
            for a_dash in model.A
        )

    model.transition = Constraint(model.N, model.S, rule=transition_rule)

    # Solve the model using Gurobi
    solver = SolverFactory("gurobi")
    result = solver.solve(model, tee=True)

    # Check the solution status and extract the solution
    optimal_policy = np.zeros((n_arms, n_state, n_action))
    if (
        result.solver.status == SolverStatus.ok
        and result.solver.termination_condition == TerminationCondition.optimal
    ):
        for n in model.N:
            for s in model.S:
                for a in model.A:
                    optimal_policy[n, s, a] = model.w[n, s, a].value
        opt_value = value(model.objective)
        return opt_value, optimal_policy
    else:
        raise Exception("No feasible solution :(")
