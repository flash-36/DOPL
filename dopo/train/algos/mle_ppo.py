import numpy as np
from scipy.optimize import minimize
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from dopo.utils import compute_ELP_pyomo, wandb_log_latest
from dopo.registry import register_training_function
from dopo.utils.neural_network import PPOAgent
import torch
from dopo.train.algos.mle_lp import mle_bradley_terry


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@register_training_function("mle_ppo")
def train(env, cfg):
    K = cfg["K"]
    clip_coeff = cfg["clip_coeff"]
    lr = cfg["lr"]
    gamma = cfg["gamma"]
    gae_lambda = cfg["gae_lambda"]
    agent = PPOAgent(
        len(env.observation_space.nvec),
        cfg["nn_size"],
        len(env.map_combinatorial_to_binary),
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=0.1, total_iters=K
    )

    # Store traj data
    states = torch.zeros((env.H,) + env.observation_space.shape).to(device)
    actions = torch.zeros((env.H)).to(device)
    logprobs = torch.zeros((env.H)).to(device)
    rewards = torch.zeros((env.H)).to(device)
    values = torch.zeros((env.H)).to(device)

    metrics = {"reward": [], "R_error": [], "loss": []}

    num_arms = len(env.P_list)
    num_states = env.P_list[0].shape[0]
    R_true = np.array(env.R_list)[:, :, 0]
    R_est = np.ones((num_arms, num_states)) * 0.5
    for k in tqdm(range(K)):
        battle_data = []
        s_list = env.reset()
        reward_episode = 0
        for t in range(env.H):
            with torch.no_grad():
                s = torch.tensor(s_list, dtype=torch.float32).to(device)
                action, log_prob, _, value = agent.get_action_and_value(s)
            s_list, reward, _, _, info = env.step(
                env.map_combinatorial_to_binary[action.item()]
            )
            # Update battle data to fit Bradley-Terry model using MLE
            for record in info["duelling_results"]:
                winner, loser = record
                battle_data.append([winner, s_list[winner], loser, s_list[loser]])

            states[t] = s
            actions[t] = action
            logprobs[t] = log_prob
            values[t] = value

            reward_episode += reward

        R_est = mle_bradley_terry(np.array(battle_data), R_est)
        for t in range(env.H):
            rewards[t] = sum(R_est[np.arange(num_arms), states[t].long().cpu().numpy()])
        loss = ppo_update(
            agent,
            optimizer,
            scheduler,
            states,
            actions,
            logprobs,
            rewards,
            values,
            gamma,
            clip_coeff,
            gae_lambda,
            env,
        )

        metrics["R_error"].append(np.linalg.norm(R_est - R_true))
        metrics["reward"].append(reward_episode)
        metrics["loss"].append(loss)
        wandb_log_latest(metrics)
    return metrics


def ppo_update(
    agent,
    optimizer,
    scheduler,
    states,
    actions,
    logprobs,
    rewards,
    values,
    gamma,
    clip_coeff,
    gae_lambda,
    env,
):
    returns = compute_returns(rewards, values, gamma, gae_lambda)
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    # flatten the batch
    b_obs = states.reshape((-1,) + env.observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape(
        -1,
    )
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(env.H)
    clipfracs = []
    loss_total = 0
    for epoch in range(4):  # ppo_epochs = 4
        np.random.shuffle(b_inds)
        for start in range(0, env.H, env.H):
            end = start + env.H
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                b_obs[mb_inds], b_actions.long()[mb_inds]
            )
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > clip_coeff).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - clip_coeff, 1 + clip_coeff
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if True:  # Always clip value loss
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -clip_coeff,
                    clip_coeff,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - 0.01 * entropy_loss + v_loss * 0.5

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            loss_total += loss.item()
    return loss_total


def compute_returns(rewards, values, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards).to(device)
    lastgaelam = 0
    for t in reversed(range(len(rewards) - 1)):
        nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * lastgaelam
    returns = advantages + values
    return returns
