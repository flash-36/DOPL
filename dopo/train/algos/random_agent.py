from tqdm import tqdm
from dopo.utils import wandb_log_latest
from dopo.registry import register_training_function
from dopo.train.helpers import apply_index_policy
import numpy as np



@register_training_function("random")
def train(env, cfg):

    K = cfg["K"]

    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]

    metrics = {
        "reward": [],
    }

    for k in tqdm(range(K)):
        # Start rollout using  random index policy
        s_list = env.reset()
        reward_episode = 0
        for t in range(env.H):
            random_index_matrix = np.random.rand(num_arms, num_states)
            action = apply_index_policy(s_list, random_index_matrix, env.arm_constraint)
            s_dash_list, reward, _, _, info = env.step(action)
            reward_episode += reward
            s_list = s_dash_list
        metrics["reward"].append(reward_episode)
        wandb_log_latest(metrics, "random")
    return metrics
