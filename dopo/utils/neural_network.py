import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return self.softmax(out)


def initialize_policy_net(env, cfg):
    hidden_size = cfg["nn_size"]
    input_size = output_size = len(env.P_list)
    policy_net = SimpleNeuralNet(input_size, hidden_size, output_size)
    return policy_net
