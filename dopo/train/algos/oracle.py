from tqdm import tqdm
from dopo.utils import wandb_log_latest
from dopo.registry import register_training_function
from dopo.train.helpers import apply_index_policy


@register_training_function("oracle")
def train(env, cfg):
    K = cfg["K"]
    metrics = {
        "reward": [],
    }
    for k in tqdm(range(K)):
        # Start rollout using  optimal index policy
        s_list = env.reset()
        reward_episode = 0
        for t in range(env.H):
            action = apply_index_policy(s_list, env.opt_index, env.arm_constraint)
            s_dash_list, reward, _, _, info = env.step(action)
            reward_episode += reward
            s_list = s_dash_list
        metrics["reward"].append(reward_episode)
        wandb_log_latest(metrics, "oracle")
    return metrics
