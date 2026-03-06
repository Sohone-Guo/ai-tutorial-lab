import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    """
    最简单策略网络：
    输入：环境状态 obs
    输出：每个动作的概率
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


def compute_returns(rewards, gamma=0.99):
    """
    计算每一步的折扣回报 G_t
    """
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)

    # 标准化，训练更稳
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


def train():
    # 创建环境
    env = gym.make("CartPole-v1")

    # 读取状态维度、动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 定义策略网络
    policy = PolicyNet(state_dim, action_dim)

    # 优化器
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # 训练参数
    num_episodes = 500
    gamma = 0.99

    for episode in range(num_episodes):
        # Gymnasium: reset() -> (obs, info)
        state, info = env.reset()

        log_probs = []
        rewards = []
        episode_reward = 0

        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # 输出动作概率
            probs = policy(state_tensor)
            dist = Categorical(probs)

            # 采样动作
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Gymnasium: step() -> (obs, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            episode_reward += reward

            state = next_state

        # 计算回报
        returns = compute_returns(rewards, gamma)

        # REINFORCE 损失
        loss = []
        for log_prob, G in zip(log_probs, returns):
            loss.append(-log_prob * G)
        loss = torch.stack(loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:4d} | Reward: {episode_reward:.1f} | Loss: {loss.item():.4f}")

    env.close()
    torch.save(policy.state_dict(), "policy_cartpole.pth")
    print("训练完成，模型已保存到 policy_cartpole.pth")


def test():
    env = gym.make("CartPole-v1", render_mode="human")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNet(state_dim, action_dim)
    policy.load_state_dict(torch.load("policy_cartpole.pth"))
    policy.eval()

    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            probs = policy(state_tensor)
            action = torch.argmax(probs, dim=-1).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        state = next_state

    env.close()
    print("测试回合总奖励:", total_reward)


if __name__ == "__main__":
    train()
    # 训练完以后再打开下面这行测试
    # test()