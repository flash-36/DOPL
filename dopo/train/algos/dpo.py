import torch.nn.functional as F
import torch
from dopo.utils import initialize_policy_net
from tqdm import tqdm
from dopo.utils import (
    wandb_log_latest,
)
from dopo.train.helpers import apply_nn_policy
from dopo.registry import register_training_function
import logging

log = logging.getLogger(__name__)


@register_training_function("dpo")
def train(env, cfg):
    K = cfg["K"]
    ref_update_freq = cfg["ref_update_freq"]
    beta = cfg["beta"]  # TODO: beta scheduler
    policy_net = initialize_policy_net(env, cfg)
    policy_net_ref = initialize_policy_net(env, cfg)
    metrics = {
        "reward": [],
        "dpo_loss": [],
    }
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=cfg["lr"])
    for k in tqdm(range(K)):
        # Start rollout using index policy
        if k % ref_update_freq == 0 and k != 0:
            policy_net_ref.load_state_dict(policy_net_buffer.state_dict())
        s_list = env.reset()
        reward_episode = 0
        loss_episode = 0
        for t in range(env.H):
            action, action_probs = apply_nn_policy(
                s_list, policy_net, env.arm_constraint
            )
            _, action_probs_ref = apply_nn_policy(
                s_list, policy_net_ref, env.arm_constraint
            )
            s_dash_list, reward, _, _, info = env.step(action)
            reward_episode += reward

            # Update F_estimate
            for record in info["duelling_results"]:
                winner, loser = record
                pi_log_ratio = torch.log(action_probs[winner]) - torch.log(
                    action_probs[loser]
                )
                ref_log_ratio = torch.log(action_probs_ref[winner]) - torch.log(
                    action_probs_ref[loser]
                )
                loss_episode += -F.logsigmoid(beta * (pi_log_ratio - ref_log_ratio))

            s_list = s_dash_list

        policy_net_buffer = policy_net
        optimizer.zero_grad()
        loss_episode.backward()
        optimizer.step()
        metrics["reward"].append(reward_episode)
        metrics["dpo_loss"].append(loss_episode.item())

        wandb_log_latest(metrics, "dpo")

    return metrics
