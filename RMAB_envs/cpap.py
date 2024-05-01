#cpap environment

import numpy as np
import gymnasium as gym


n_states = 3
n_actions = 2

#arm 1 transition matrix for action 0 p10 refers to arm 1 action 0

p10 = np.zeros((n_states,n_states))

p10[0][0] = 0.0385
p10[0][1] = 0
p10[0][2] = 0.9615
p10[1][1] = 0
p10[1][0] = 0
p10[1][2] = 1
p10[2][0] = 0.0257
p10[2][1] = 0.0245
p10[2][2] = 0.9498


#arm1 transition matrix for action 1

p11 = np.zeros((n_states,n_states))
p11[0][0] =0
p11[1][1] = 0
p11[2][2] =1
p11[0][1] =0
p11[1][0]=0
p11[1][2]=1
p11[0][2]=1
p11[2][0]=0
p11[2][1]=0


#arm2 transition matrix for action 0
p20 = np.zeros((n_states,n_states))

p20[0][0] = 0.7424
p20[0][1] = 0.0741
p20[0][2] = 0.1835
p20[1][1] = 0.1634
p20[1][0] = 0.3399
p20[1][2] = 0.4967
p20[2][0] = 0.2323
p20[2][1] = 0.1020
p20[2][2] = 0.6657

#arm2 transition matrix for action 1

p21 = np.zeros((n_states,n_states))

p21[0][0] = 0.1424
p21[0][1] = 0.3741
p21[0][2] = 0.4835
p21[1][1] = 0
p21[1][0] = 0.1399
p21[1][2] = 0.8601
p21[2][0] = 0.0323
p21[2][1] = 0
p21[2][2] = 0.9677


P = np.zeros((2, 3, 3, 2))
P[0, :, :, 0] = p10 # Arm 1, Action 0
P[0, :, :, 1] = p11 # Arm 1, Action 1
P[1, :, :, 0] = p20 # Arm 2, Action 0
P[1, :, :, 1] = p21 # Arm 2, Action 1

print("The transition matrix for arm 1 action 0 is",P[0,:,:,0])
print("The transition matrix for arm 1 action 0 is",P[0,:,:,1])
print("The transition matrix for arm 1 action 0 is",P[1,:,:,0])
print("The transition matrix for arm 1 action 0 is",P[1,:,:,1])


for arm_index in range(P.shape[0]):
    # Loop over actions (fourth dimension of P)
    for action_index in range(P.shape[3]):
        # Get the specific sub-matrix for this arm and action
        transition_matrix = P[arm_index, :, :, action_index]
        
        # Calculate the sum of each row in this sub-matrix
        row_sums = np.sum(transition_matrix, axis=1)
        
        # Check if all row sums are close to 1 (using a tolerance, since we are dealing with floating point arithmetic)
        if np.allclose(row_sums, np.ones((P.shape[1],)), atol=1e-8):
            print(f"Arm {arm_index + 1}, Action {action_index + 1}: Rows sum to 1 - OK")
        else:
            print(f"Arm {arm_index + 1}, Action {action_index + 1}: Row sums not 1 - ERROR")
            print("Row sums:", row_sums)  # Print the row sums for inspection
np.save('cpap.npy', P) 


