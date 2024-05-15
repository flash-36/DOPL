import numpy as np

class Wireless_Scheduling:
    def __init__(self, num_channels,max_data,num_states,num_actions):
        """
        num_channels = 2 : good:1 or bad: 0
        max_data: 100 mb. The maximum data that can be transmitted
        num_states = 2*100
        num_actions = 2 i.e 0 or 1
        """
        self.num_channels = num_channels
        self.max_data = max_data
        self.num_states = num_states
        self.num_actions = num_actions
        self.R = np.zeros((num_states,num_actions))
        self.P = np.zeros((num_states,num_states,num_actions))
        self.flip_good = np.random.random()
        self.flip_bad = np.random.random()
        self.big_transition_good = 0.8
        self.small_transition_good = 0.2
        self.small_transition_bad = 0.8
        self.big_transition_bad = 0.2


    def populate_reward(self):
        
        for channel in range(self.num_channels):
            for data in range(1,self.max_data+1):
                for action in range(self.num_actions):
                    state_index = channel * (self.max_data)+data
                    self.R[state_index-1][action] = (self.max_data+1-data)/(self.max_data)



    def populate_transition_matrix(self):

        for channel in range(self.num_channels):
            for data in range(1,self.max_data+1):
                for action in range(self.num_actions):

                    if(action==0 and channel == 1):
                        #Already in good channel state
                        #in that case with flip probability the channel state might change
                        state_index_bad = (1 - channel) * (self.max_data)+data-1
                        state_index_good = channel * (self.max_data)+data-1
                        self.P[state_index_good,state_index_good,action] = 1-self.flip_good
                        self.P[state_index_good,state_index_bad,action] = self.flip_good

                    elif(action==0 and channel == 0):
                        #arm in a bad state
                        state_index_bad = channel * (self.max_data)+data-1
                        state_index_good = (1-channel) * (self.max_data)+data-1
                        self.P[state_index_bad,state_index_good,action] = self.flip_bad
                        self.P[state_index_bad,state_index_bad,action] = 1 - self.flip_bad

                    elif(action==1 and channel == 1):
                        state_index = (channel) * (self.max_data)+data-1
                        if(data>2):
                            data1 = data -1
                            data2 = data -2
                            state_index_1 = (1-channel) * (self.max_data)+data1-1
                            state_index_2 = (1-channel) * (self.max_data)+data2-1
                            state_index_3 = (channel) * (self.max_data)+data1-1
                            state_index_4 = (channel) * (self.max_data)+data2-1

                            self.P[state_index,state_index_1,action] += (self.flip_good)*self.small_transition_good
                            self.P[state_index,state_index_2,action] += (self.flip_good)*self.big_transition_good
                            self.P[state_index,state_index_3,action] += (1 - self.flip_good)*self.small_transition_good
                            self.P[state_index,state_index_4,action] += (1 - self.flip_good)*self.big_transition_good

                        elif(data==2):
                            state_index_1 = (1-channel) * (self.max_data)
                            state_index_2 = (channel) * (self.max_data)
                            self.P[state_index,state_index_1,action] = self.flip_good*(self.small_transition_good)
                            self.P[state_index,state_index_2,action] = (1-self.flip_good)*(self.small_transition_good)
                            for data_next in range(1,self.max_data+1):
                                state_index_1 = (1-channel) * (self.max_data)+ data_next-1
                                state_index_2 = (channel) * (self.max_data)+ data_next-1
                                self.P[state_index,state_index_1,action] += self.flip_good*(self.big_transition_good)/self.num_states*2
                                self.P[state_index,state_index_2,action] += (1-self.flip_good)*(self.big_transition_good)/self.num_states*2

                        elif(data==1):
                            for data_next in range(1,self.max_data+1):
                                state_index_1 = (1-channel) * (self.max_data)+ data_next-1
                                state_index_2 = (channel) * (self.max_data)+ data_next-1
                                self.P[state_index,state_index_1,action] += self.flip_good/self.num_states*2
                                self.P[state_index,state_index_2,action] += (1-self.flip_good)/self.num_states*2


                    elif(action==1 and channel == 0):
                        state_index = (channel) * (self.max_data)+data-1
                        if(data>=2):
                            data1 = data 
                            data2 = data - 1
                            state_index_1 = (1-channel) * (self.max_data)+data1-1
                            state_index_2 = (1-channel) * (self.max_data)+data2-1
                            state_index_3 = (channel) * (self.max_data)+data1-1
                            state_index_4 = (channel) * (self.max_data)+data2-1

                            self.P[state_index,state_index_1,action] += (self.flip_bad)*self.small_transition_bad
                            self.P[state_index,state_index_2,action] += (self.flip_bad)*self.big_transition_bad
                            self.P[state_index,state_index_3,action] += (1 - self.flip_bad)*self.small_transition_bad
                            self.P[state_index,state_index_4,action] += (1 - self.flip_bad)*self.big_transition_bad

                        elif(data==1):
                            state_index_1 = (1-channel) * (self.max_data)
                            state_index_2 = (channel) * (self.max_data)
                            self.P[state_index,state_index_1,action] += self.flip_bad*(self.small_transition_bad)
                            self.P[state_index,state_index_2,action] += (1-self.flip_bad)*(self.small_transition_bad)
                            for data_next in range(1,self.max_data+1):
                                state_index_1 = (1-channel) * (self.max_data)+ data_next-1
                                state_index_2 = (channel) * (self.max_data)+ data_next-1
                                left_probability = 1 - self.flip_bad*(self.small_transition_bad) - (1-self.flip_bad)*(self.small_transition_bad)
                                self.P[state_index,state_index_1,action] += left_probability/self.num_states
                                self.P[state_index,state_index_2,action] += left_probability/self.num_states

def save_matrices(P, R, arm_type):
    np.save(f"wireless_scheduling_{arm_type}_transitions.npy", P)
    np.save(f"wireless_scheduling_{arm_type}_rewards.npy", R)
    print(f"Matrices for {arm_type} saved successfully.")


def main():
    ws = Wireless_Scheduling(2, 3, 6, 2)
    ws.populate_reward()
    ws.populate_transition_matrix()
    for action in range(ws.num_actions):
        print(f"Transition Matrix for Action {action}:")
        print(ws.P[:, :, action])
        print("\n")  # Adds a new line for better separation between matrices
        print("THE SHAPE OF PROBABILITY MATRIX IS ",ws.P.shape)
        print("THE SHAPE OF reward MATRIX IS ",ws.R.shape)

    # Check the sum of probabilities for each state and action to ensure it's 1 or close to 1
    prob_check = np.allclose(ws.P.sum(axis=1), 1, atol=0.01)  # Allow some tolerance

    print("Probability Check Passed:", prob_check)


    save_matrices(ws.P, ws.R, "arm_type_1")

if __name__ == "__main__":
    main()


                        

                        


        
                            







