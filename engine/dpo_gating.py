import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Iterable, Tuple, List


class DPOGatingNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 34,
        hidden_dim: int = 64,
        num_actions: int = 3,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        prefer_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda") if (device is None and prefer_cuda) else (device or torch.device("cpu"))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        try:
            self.to(self.device)
        except RuntimeError:
            self.device = torch.device("cpu")
            self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(self.device).float()
        return self.net(x)


class DPOTrainer:
    def __init__(
        self,
        model: Optional[DPOGatingNet] = None,
        lr: float = 1e-3,
        beta: float = 0.1,
        cost_beta: float = 0.1,
        action_costs: Optional[Tuple[float, float, float]] = (0.0, 0.2, 0.3),
        device: Optional[torch.device] = None,
    ) -> None:
        prefer_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda") if (device is None and prefer_cuda) else (device or torch.device("cpu"))
        try:
            self.model = model if model is not None else DPOGatingNet(device=self.device)
        except RuntimeError:
            self.device = torch.device("cpu")
            self.model = model if model is not None else DPOGatingNet(device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.beta = float(beta)
        self.cost_beta = float(cost_beta)
        self.action_costs = action_costs if action_costs is not None else (0.0, 0.2, 0.3)

    def compute_loss(self, state: torch.Tensor, winner_idx: int, loser_idx: int) -> float:
        logits = self.model.forward(state).squeeze(0)
        win = logits[int(winner_idx)]
        lose = logits[int(loser_idx)]
        cw = float(self.action_costs[int(winner_idx)])
        cl = float(self.action_costs[int(loser_idx)])
        margin_eff = self.beta * (win - lose) - self.cost_beta * (cw - cl)
        loss = F.softplus(-margin_eff)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def predict_action(self, state: torch.Tensor, use_cost_bias: bool = True) -> int:
        with torch.no_grad():
            logits = self.model.forward(state).squeeze(0)
            if use_cost_bias:
                costs = torch.tensor(self.action_costs, dtype=logits.dtype, device=logits.device)
                logits = logits - self.cost_beta * costs
            return int(torch.argmax(logits).item())


def prepare_preference_data(
    results_vimn: Iterable[dict],
    results_memento: Iterable[dict],
) -> List[Tuple[torch.Tensor, int, int]]:
    by_id_v = {r["id"]: r for r in results_vimn}
    by_id_m = {r["id"]: r for r in results_memento}
    common = set(by_id_v.keys()) & set(by_id_m.keys())
    pairs: List[Tuple[torch.Tensor, int, int]] = []
    for sid in common:
        rv = by_id_v[sid]
        rm = by_id_m[sid]
        sv = rv["state"]
        if not isinstance(sv, torch.Tensor):
            sv = torch.tensor(sv, dtype=torch.float32)
        cv = bool(rv.get("is_correct", False))
        cm = bool(rm.get("is_correct", False))
        if cv and not cm:
            pairs.append((sv, 0, 1))
        elif cm and not cv:
            pairs.append((sv, 1, 0))
    return pairs