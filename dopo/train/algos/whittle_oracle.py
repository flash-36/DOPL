from tqdm import tqdm
from dopo.utils import wandb_log_latest
from dopo.registry import register_training_function
from dopo.train.helpers import apply_index_policy
import time


@register_training_function("whittle_oracle")
def train(env, cfg):
    start_time = time.time()
    K = cfg["K"]
    metrics = {
        "reward": [],
        "run_time": 0,
    }
    for k in tqdm(range(K)):
        # Start rollout using  optimal index policy
        s_list = env.reset()
        reward_episode = 0
        for t in range(env.H):
            action = apply_index_policy(s_list, env.whittle_indices, env.arm_constraint)
            s_dash_list, reward, _, _, info = env.step(action)
            reward_episode += reward
            s_list = s_dash_list
        metrics["reward"].append(reward_episode)
        metrics["run_time"] = time.time() - start_time
        wandb_log_latest(metrics)
    return metrics
