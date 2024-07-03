import numpy as np
from dopo.utils import initialize_policy_net
import numpy as np
from tqdm import tqdm
from dopo.utils import (
    wandb_log_latest,
)
from dopo.train.helpers import apply_nn_policy
from dopo.registry import register_training_function


@register_training_function("dpo")
def train(env, cfg):
    K = cfg["K"]
    policy_net = initialize_policy_net(env, cfg)
    metrics = {
        "reward": [],
    }

    for k in tqdm(range(K)):

        # Start rollout using index policy
        s_list = env.reset()
        reward_episode = 0
        for t in range(env.H):
            action = apply_nn_policy(s_list, policy_net, env.arm_constraint)
            s_dash_list, reward, _, _, info = env.step(action)
            reward_episode += reward
            # Update F_estimate
            for record in info["duelling_results"]:
                winner, loser = record
                ## Do stuff ##

                ##############

            s_list = s_dash_list

        metrics["reward"].append(reward_episode)

        wandb_log_latest(metrics, "dpo")

    return metrics
