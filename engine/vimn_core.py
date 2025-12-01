import torch
import torch.nn as nn


class VIMN(nn.Module):
    def __init__(self, num_pois: int, num_acts: int, embed_dim: int = 128, hidden_dim: int = 256, num_users: int = None, user_embed_dim: int = None, dropout: float = 0.0):
        super().__init__()
        self.poi_emb = nn.Embedding(num_pois, embed_dim)
        self.act_emb = nn.Embedding(num_acts, embed_dim)
        self.time_emb = nn.Embedding(24, embed_dim)
        self.user_emb = nn.Embedding(num_users, user_embed_dim) if (num_users is not None and user_embed_dim is not None) else None
        in_dim = embed_dim * 3 + (user_embed_dim if self.user_emb is not None else 0)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.translator_head = nn.Linear(hidden_dim, num_acts)

    def forward(self, poi_ids: torch.Tensor, act_ids: torch.Tensor, time_ids: torch.Tensor, h0: torch.Tensor = None, user_ids: torch.Tensor = None) -> torch.Tensor:
        p = self.poi_emb(poi_ids)
        a = self.act_emb(act_ids)
        t = self.time_emb(time_ids)
        if self.user_emb is not None and user_ids is not None:
            u = self.user_emb(user_ids).unsqueeze(1).expand(-1, p.size(1), -1)
            x = torch.cat([p, a, t, u], dim=-1)
        else:
            x = torch.cat([p, a, t], dim=-1)
        x = self.input_proj(x)
        x = self.dropout(x)
        if h0 is not None and h0.dim() == 2:
            h0 = h0.unsqueeze(0)
        out, h = self.gru(x, h0)
        ht = out[:, -1, :]
        return ht

    def step(self, poi_idx: int, act_idx: int, time_hour: int, h_prev: torch.Tensor = None, user_idx: int = None) -> torch.Tensor:
        device = next(self.parameters()).device
        poi = torch.tensor([[poi_idx]], dtype=torch.long, device=device)
        act = torch.tensor([[act_idx]], dtype=torch.long, device=device)
        tim = torch.tensor([[int(time_hour) % 24]], dtype=torch.long, device=device)
        p = self.poi_emb(poi)
        a = self.act_emb(act)
        t = self.time_emb(tim)
        if self.user_emb is not None and user_idx is not None:
            u = self.user_emb(torch.tensor([user_idx], dtype=torch.long, device=device)).unsqueeze(1)
            x = torch.cat([p, a, t, u], dim=-1)
        else:
            x = torch.cat([p, a, t], dim=-1)
        x = self.input_proj(x)
        x = self.dropout(x)
        if h_prev is not None and h_prev.dim() == 2:
            h_prev = h_prev.unsqueeze(0)
        out, h = self.gru(x, h_prev)
        ht = out[:, -1, :]
        return ht

    def translate(self, h: torch.Tensor) -> torch.Tensor:
        return self.translator_head(h)

    def forward_with_logits(self, poi_ids: torch.Tensor, act_ids: torch.Tensor, time_ids: torch.Tensor, h0: torch.Tensor = None):
        ht = self.forward(poi_ids, act_ids, time_ids, h0)
        logits = self.translate(ht)
        return ht, logits