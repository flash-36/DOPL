import numpy as np
import math
import time
import pulp as p
import sys



class Optimal:
    def __init__(self,reward,budget,P,n_arms,n_state,n_action):
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

        optimal_policy = np.zeros((self.n_arms,self.n_state,self.n_action))
        prob = p.LpProblem("OptimalLP",p.LpMaximize)
        p_keys = [(n,s,a) for n in range(self.n_arms) for s in range(self.n_state) for a in range(self.n_action)] 
        w = p.LpVariable.dicts("w",p_keys,lowBound=0,upBound=1,cat='continous')

        r = {}
        for n in range(self.n_arms):
            r[n] = {}

            for state in range(self.n_state):
                r[n][state] = self.R[n][state]
        #objective function
        opt_prob+= p.lpSum([w[(n,s,a)]*r[n][s] for n in range(self.n_arms) for s in range(self.n_state) for a in range(self.n_action)])
        #Budget Constraint
        opt_prob+= p.lpSum([w[(n,s,a)]*a for n in range(self.n_arms) for s in range(self.n_state) for a in range(self.n_action)])-self.budge<=0

        for n in range(self.arms):
            w_list = [w[(n,s,a)] for s in range(self.state) for a in range(self.n_action)]
            opt_prob+= p.lpSum(w_list) -1 == 0

        for n in range(self.arms):
            for s in range(self.n_state):
                for a in range(self.actions):
                    opt_prob+= w[(n,s,a)] >=0


        for n in range(self.arms):
            for s in range(self.n_state):
                a_list = [w[(n,s,a)] for a in range(self.n_action)]
                b_list = [w[(n,s_dash,a_dash)]*self.P[s_dash][a_dash][s] for s_dash in range(self.n_state) for a_dash in range(self.n_action)]
                opt_prob+= p.lpSum(a_list) - p.lpSum(b_list) == 0

        status = opt_prob.solve(p.PULP_CBC_CMD(fracGap=0.01,msg = 0))
        if p.LpStatus[status]!= 'optimal':
            print("Optimal policy calculation failed")
            return optimal_policy
        


        for n in range(self.n_arms):
            for s in range(self.n_state):
                for a in range(self.n_action):
                    optimal_policy[n,s,a] = w[(n,s,a)].varValue
                    if(optimal_policy[n,s,a]) < 0  and optimal_policy[n,s,a]> -0.001:
                        optimal_policy[n,s,a] = 0
                    elif optimal_policy[n,s,a] < -0.001:
                        print("Invalid Value")
                        sys.exit()


        return optimal_policy
