import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional


class MementoPolicyNet(nn.Module):
    """
    Learnable scoring network for long-term memory selection.

    Input shape:
    - x: Tensor of shape (B, input_dim) where x = concat(state_emb, memory_emb)

    Output:
    - score: Tensor of shape (B, 1), in [0, 1], representing relevance/utility.
    """

    def __init__(self, input_dim: int = 1536 * 2, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        y = self.fc2(h)
        y = self.sigmoid(y)
        return y


class MementoTrainer:
    """
    Trainer for MementoPolicyNet.
    Provides train_batch and inference for ranking.
    """

    def __init__(self, input_dim: int = 1536 * 2, hidden_dim: int = 512, lr: float = 1e-3):
        self.model = MementoPolicyNet(input_dim=input_dim, hidden_dim=hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

    def train_batch(
        self,
        state_embs: torch.Tensor,
        memory_embs: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, torch.Tensor]:
        """
        Standard PyTorch training step.

        Inputs:
        - state_embs: (B, D) current state embeddings
        - memory_embs: (B, D) candidate memory embeddings (aligned with state_embs)
        - labels: (B, 1) binary labels in {0.0, 1.0}

        Returns:
        - loss_value (float)
        - scores: (B, 1)
        """
        self.model.train()
        x = torch.cat([state_embs, memory_embs], dim=-1)
        scores = self.model(x)
        loss = self.criterion(scores, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item()), scores.detach()

    @torch.no_grad()
    def predict_scores(
        self,
        state_emb: torch.Tensor,
        candidate_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference for ranking.

        Inputs:
        - state_emb: (D,) or (1, D)
        - candidate_embs: (K, D)

        Output:
        - scores: (K, 1) relevance scores in [0, 1]
        """
        self.model.eval()
        if state_emb.dim() == 1:
            state_emb = state_emb.unsqueeze(0)
        K = candidate_embs.shape[0]
        state_expand = state_emb.expand(K, -1)
        x = torch.cat([state_expand, candidate_embs], dim=-1)
        scores = self.model(x)
        return scores


def construct_training_samples(
    current_ground_truth_act: Optional[str],
    current_ground_truth_poi: Optional[int],
    candidate_memories: List[dict],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build training tensors (memory_embs_tensor, labels_tensor).

    Inputs:
    - current_ground_truth_act: str or None
    - current_ground_truth_poi: int or None
    - candidate_memories: list of dicts, each dict must include:
        {
            'memory_emb': torch.Tensor of shape (D,),
            'act_name': str,
            'poi_id': int,
        }

    Labeling rule:
    - label = 1.0 if act_name == current_ground_truth_act OR poi_id == current_ground_truth_poi
    - else label = 0.0

    Returns:
    - memory_embs: (K, D)
    - labels: (K, 1)
    """
    emb_list: List[torch.Tensor] = []
    lab_list: List[float] = []
    for mem in candidate_memories:
        emb = mem.get('memory_emb')
        act = mem.get('act_name')
        pid = mem.get('poi_id')
        if emb is None:
            continue
        emb_list.append(emb)
        is_pos = False
        if current_ground_truth_act is not None and act is not None:
            is_pos = is_pos or (str(act) == str(current_ground_truth_act))
        if current_ground_truth_poi is not None and pid is not None:
            is_pos = is_pos or (int(pid) == int(current_ground_truth_poi))
        lab_list.append(1.0 if is_pos else 0.0)

    if len(emb_list) == 0:
        return torch.empty((0, 0)), torch.empty((0, 1))

    memory_embs = torch.stack(emb_list, dim=0)
    labels = torch.tensor(lab_list, dtype=torch.float32).unsqueeze(-1)
    return memory_embs, labels