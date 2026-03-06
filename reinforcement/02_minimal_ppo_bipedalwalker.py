# 这个是训练机器人行走的简单实例。
# pip install "gymnasium[box2d]" torch numpy
import math
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# =========================
# 1. Actor-Critic 网络
# =========================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()

        # actor: 输出动作均值
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )

        # 可学习的动作标准差（对每个动作维度一个参数）
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # critic: 输出状态价值 V(s)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def get_dist(self, obs):
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def get_action_and_value(self, obs):
        """
        给定状态，采样动作，并返回：
        action, log_prob, entropy, value
        """
        dist = self.get_dist(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value

    def evaluate_actions(self, obs, actions):
        """
        PPO更新时，重新计算新策略下的 log_prob / entropy / value
        """
        dist = self.get_dist(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value


# =========================
# 2. Rollout Buffer
# =========================
class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, obs, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.__init__()


# =========================
# 3. 计算 GAE 和 returns
# =========================
def compute_gae(rewards, dones, values, next_value, gamma=0.99, gae_lambda=0.95):
    """
    rewards: [T]
    dones:   [T]
    values:  [T]
    next_value: 标量，最后一个状态的 V(s_T+1)
    """
    advantages = []
    gae = 0.0

    values = values + [next_value]

    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)

    returns = [adv + v for adv, v in zip(advantages, values[:-1])]
    return advantages, returns


# =========================
# 4. PPO 更新
# =========================
def ppo_update(
    model,
    optimizer,
    obs_tensor,
    actions_tensor,
    old_log_probs_tensor,
    returns_tensor,
    advantages_tensor,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.001,
    epochs=10,
    batch_size=64,
):
    n = obs_tensor.size(0)

    # advantage 标准化，实践里很常见
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
        advantages_tensor.std() + 1e-8
    )

    for _ in range(epochs):
        indices = np.arange(n)
        np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            batch_obs = obs_tensor[batch_idx]
            batch_actions = actions_tensor[batch_idx]
            batch_old_log_probs = old_log_probs_tensor[batch_idx]
            batch_returns = returns_tensor[batch_idx]
            batch_advantages = advantages_tensor[batch_idx]

            new_log_probs, entropy, values = model.evaluate_actions(batch_obs, batch_actions)

            # PPO ratio
            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            # PPO clip objective
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # critic loss
            critic_loss = ((batch_returns - values) ** 2).mean()

            # entropy bonus
            entropy_loss = entropy.mean()

            loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()


# =========================
# 5. 主训练循环
# =========================
def train():
    env = gym.make("BipedalWalker-v3")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim, hidden_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    num_updates = 300          # 更新次数
    rollout_steps = 2048       # 每次先收集多少步数据
    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2

    buffer = RolloutBuffer()

    obs, info = env.reset()
    episode_reward = 0.0
    episode_count = 0

    for update in range(1, num_updates + 1):
        buffer.clear()

        # ========== 收集一批 rollout ==========
        for step in range(rollout_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action_tensor, log_prob_tensor, entropy_tensor, value_tensor = model.get_action_and_value(obs_tensor)

            action = action_tensor.squeeze(0).numpy()

            # BipedalWalker 动作范围是 [-1, 1]，这里手动裁剪
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                done=float(done),
                log_prob=log_prob_tensor.item(),
                value=value_tensor.item(),
            )

            obs = next_obs
            episode_reward += reward

            if done:
                episode_count += 1
                print(f"Update {update:3d} | Episode {episode_count:4d} | Reward = {episode_reward:.2f}")
                obs, info = env.reset()
                episode_reward = 0.0

        # ========== 计算最后一个状态的 value ==========
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            next_value = model.critic(obs_tensor).item()

        advantages, returns = compute_gae(
            rewards=buffer.rewards,
            dones=buffer.dones,
            values=buffer.values,
            next_value=next_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # ========== 转 tensor ==========
        obs_tensor = torch.tensor(np.array(buffer.obs), dtype=torch.float32)
        actions_tensor = torch.tensor(np.array(buffer.actions), dtype=torch.float32)
        old_log_probs_tensor = torch.tensor(buffer.log_probs, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)

        # ========== PPO 更新 ==========
        ppo_update(
            model=model,
            optimizer=optimizer,
            obs_tensor=obs_tensor,
            actions_tensor=actions_tensor,
            old_log_probs_tensor=old_log_probs_tensor,
            returns_tensor=returns_tensor,
            advantages_tensor=advantages_tensor,
            clip_eps=clip_eps,
            value_coef=0.5,
            entropy_coef=0.001,
            epochs=10,
            batch_size=64,
        )

        if update % 10 == 0:
            torch.save(model.state_dict(), "ppo_bipedalwalker_minimal.pth")
            print(f"模型已保存: ppo_bipedalwalker_minimal.pth")

    env.close()


def test():
    env = gym.make("BipedalWalker-v3", render_mode="human")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim, hidden_dim=128)
    model.load_state_dict(torch.load("ppo_bipedalwalker_minimal.pth", map_location="cpu"))
    model.eval()

    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            # 测试时用均值动作，更稳定
            mean = model.actor(obs_tensor)
            action = mean.squeeze(0).numpy()

        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print("Test reward:", total_reward)
    env.close()


if __name__ == "__main__":
    train()
    # test()