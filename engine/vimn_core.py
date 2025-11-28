import torch
import torch.nn as nn


class VIMN(nn.Module):
    def __init__(self, num_pois: int, num_acts: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.poi_emb = nn.Embedding(num_pois, embed_dim)
        self.act_emb = nn.Embedding(num_acts, embed_dim)
        self.time_emb = nn.Embedding(24, embed_dim)
        self.input_proj = nn.Linear(embed_dim * 3, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.translator_head = nn.Linear(hidden_dim, num_acts)

    def forward(self, poi_ids: torch.Tensor, act_ids: torch.Tensor, time_ids: torch.Tensor, h0: torch.Tensor = None) -> torch.Tensor:
        p = self.poi_emb(poi_ids)
        a = self.act_emb(act_ids)
        t = self.time_emb(time_ids)
        x = torch.cat([p, a, t], dim=-1)
        x = self.input_proj(x)
        if h0 is not None and h0.dim() == 2:
            h0 = h0.unsqueeze(0)
        out, h = self.gru(x, h0)
        ht = out[:, -1, :]
        return ht

    def step(self, poi_idx: int, act_idx: int, time_hour: int, h_prev: torch.Tensor = None) -> torch.Tensor:
        device = next(self.parameters()).device
        poi = torch.tensor([[poi_idx]], dtype=torch.long, device=device)
        act = torch.tensor([[act_idx]], dtype=torch.long, device=device)
        tim = torch.tensor([[int(time_hour) % 24]], dtype=torch.long, device=device)
        p = self.poi_emb(poi)
        a = self.act_emb(act)
        t = self.time_emb(tim)
        x = torch.cat([p, a, t], dim=-1)
        x = self.input_proj(x)
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