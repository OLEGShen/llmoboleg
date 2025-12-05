import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Optional, Tuple


# 该模块实现“上下文强盗（Contextual Bandit）”门控策略：
# 基于状态（VIMN 不确定性、Memento 相关性、时间嵌入），动态选择
# 直觉（VIMN）、经验（Memento）或混合（两者）三种提示构造模式。


class RLGatingNetwork(nn.Module):
    """
    强化学习门控网络（策略网络），输入状态向量，输出 3 个动作的 logits。

    输入 state: [vimn_entropy(1), memento_score(1), time_embedding(32)] 共约 34 维
    输出动作：
      0 -> 直觉模式（仅用 VIMN）
      1 -> 经验模式（仅用 Memento）
      2 -> 混合模式（两者都用）
    """

    def __init__(
        self,
        input_dim: int = 34,
        hidden_dim: int = 64,
        num_actions: int = 3,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向，返回动作 logits（未归一化）"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(self.device).float()
        return self.net(x)

    def select_action(self, state_tensor: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        使用 Categorical(logits=...) 进行带探索的采样，返回 (action_idx, log_prob)。
        """
        logits = self.forward(state_tensor)  # [1, num_actions]
        dist = Categorical(logits=logits.squeeze(0))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob


class GatingTrainer:
    """
    使用 REINFORCE（Policy Gradient）对门控网络进行训练：
    损失为 -log_prob * reward，优化器采用 Adam。
    """

    def __init__(
        self,
        input_dim: int = 34,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.policy = RLGatingNetwork(
            input_dim=input_dim, hidden_dim=hidden_dim, device=self.device
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def train_step(self, log_prob: torch.Tensor, reward: float) -> float:
        """单步 REINFORCE 更新，返回当前 loss 值"""
        reward_t = torch.tensor(reward, dtype=torch.float32, device=log_prob.device)
        loss = -log_prob * reward_t
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def calculate_reward(self, is_correct: bool, action_idx: int, beta: float = 0.1) -> float:
        """
        奖励函数（带动作成本惩罚）：
        - 基础奖励: 正确 +1.0，错误 -0.1
        - 动作成本: 直觉=0.0，经验=0.2，混合=0.3
        - 最终奖励: base - beta * cost
        """
        base = 1.0 if is_correct else -0.1
        if action_idx == 0:
            cost = 0.0
        elif action_idx == 1:
            cost = 0.2
        elif action_idx == 2:
            cost = 0.3
        else:
            cost = 0.0
        return float(base - (beta * cost))