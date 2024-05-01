import numpy as np
import math
import time
import pulp as p
import sys


# Question remaining why archana chose lp.minimize which should have been maximize

class ELP:
    def __init__(self,delta,P_hat,budget,n_state,n_actions,Reward,n_arms):
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

        index_policy = np.zeros((self.n_arms,self.n_state,self.n_state,self.n_action)) 
        opt_prob = p.LpProblem("ExtendedLP",p.LpMaximize)
        idx_p_keys = [(n,s,a,s_dash) for n in range(self.n_arms) for s in range(self.n_state) for a in range(self.n_action) for s_dash in range(self.n_state)]
        w = p.LpVariable.dicts("w",idx_p_keys,lowBound=0,upBound=1,cat='continous')

        
        r = {}
        for n in range(self.n_arms):
            r[n] = {}

            for state in range(self.n_state):
                r[n][state] = self.R[n][state]

        #objective equation
        opt_prob+= p.lpSum([w[(n,s,a,s_dash)]*r[n][s] for n in range(self.n_arms) for s in range(self.n_state) for a in range(self.n_action) for s_dash in range(self.n_state)])

        #Budget Constraint

        opt_prob+= p.lpSum([w[(n,s,a,s_dash)]*a for n in range(self.n_arms) for s in range(self.n_state) for a in range(self.n_action) for s_dash in range(self.n_state)])-self.budge<=0

        for n in range(self.n_arms):
            for s in self.range(self.n_state):
                w_list = [w[(n,s,a,s_dash)] for a in range(self.n_action) for s_dash in range(self.n_state)]
                w_1_list = [w[(n,s_dash,a_dash,s)] for a_dash in range(self.n_action) for s_dash in range(self.n_state)]
                opt_prob += p.lpSum(w_list) - p.lpSum(w_1_list) == 0


        for n in range(self.n_arms):

            a_list = [w[(n,s,a,s_dash)] for s in range(self.n_state) for a in range(self.n_action) for s_dash in range(self.n_state)] 
            opt_prob += p.lpSum(a_list) == 1

        #Extended part of the Linear Programming
        for n in range(self.n_arms):
            for s in range(self.n_state):
                for a in range(self.n_action):
                    for s_dash in range(self.n_state):

                        b_list = [w[(n,s,a,s_dash)] for s_dash in range(self.n_state)]
                        opt_prob += w[(n,s,a,s_dash)]/p.lpSum(b_list) - self.P_hat[n][s_dash][a][s] - self.delta[n][s][a] <=0

                        opt_prob += -1*w[(n,s,a,s_dash)]/p.lpSum(b_list) + self.P_hat[n][s_dash][a][s] - self.delta[n][s][a] <=0


        status = opt_prob.solve(p.PULP_CBC_CMD(fracGap=0.01,msg = 0))

        if p.LpStatus[status]!= 'optimal':
            return index_policy
        
        for n in range(self.n_arms):
            for s in range(self.n_state):
                for a in range(self.n_action):
                    for s_dash in range(self.n_state):
                        index_policy[n,s,a,s_dash] = w[(n,s,a,s_dash)].varValue
                        if(index_policy[n,s,a,s_dash]) < 0  and index_policy[n,s,a,s_dash]> -0.001:
                            index_policy[n,s,a,s_dash] = 0
                        elif index_policy[n,s,a,s_dash] < -0.001:
                            print("Invalid Value")
                            sys.exit()


        return index_policy
    

    #we need to make slight checks with the values but I am not sure what checks to be made without data








    