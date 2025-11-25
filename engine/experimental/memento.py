import torch
import torch.nn as nn
from datetime import datetime
from .vimn import DataVectorizer, VIMN_Lite, IntentContrastiveHead


class MementoCase:
    def __init__(self, traj: str, date_str: str, vec: DataVectorizer, model: VIMN_Lite, head: IntentContrastiveHead):
        self.traj = traj
        self.date = date_str
        poi, act, time = vec.vectorize_sequence([traj])
        with torch.no_grad():
            z = model(poi, act, time)
            z = head(z)
            z = nn.functional.normalize(z, dim=-1)
        self.intent = z.squeeze(0)
        items = vec._parse_route(traj)
        acts = [a for _, a, _ in items]
        hours = [h for _, _, h in items]
        self.act_set = set(acts)
        self.hour_hist = torch.zeros(24, dtype=torch.float)
        for h in hours:
            self.hour_hist[h % 24] += 1.0
        if self.hour_hist.sum() > 0:
            self.hour_hist = self.hour_hist / self.hour_hist.sum()
        self.reward = 0.0


class MementoMemory:
    def __init__(self):
        self.cases = []

    def add(self, case: MementoCase):
        self.cases.append(case)

    def update_reward(self, idx: int, r: float):
        self.cases[idx].reward = r


class MementoPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_intent = nn.Parameter(torch.tensor(1.0))
        self.w_time = nn.Parameter(torch.tensor(0.5))
        self.w_recency = nn.Parameter(torch.tensor(0.5))

    def recency_score(self, query_date: str, case_date: str):
        try:
            q = datetime.strptime(query_date, "%Y-%m-%d")
            c = datetime.strptime(case_date, "%Y-%m-%d")
            d = abs((q - c).days)
            return 1.0 / (1.0 + d / 7.0)
        except Exception:
            return 0.0

    def score(self, q_intent: torch.Tensor, q_hist: torch.Tensor, q_date: str, case: MementoCase):
        s_intent = torch.dot(q_intent, case.intent)
        s_time = torch.dot(q_hist, case.hour_hist)
        s_rec = torch.tensor(self.recency_score(q_date, case.date))
        s = self.w_intent * s_intent + self.w_time * s_time + self.w_recency * s_rec
        return s

    def select(self, q_intent: torch.Tensor, q_hist: torch.Tensor, q_date: str, memory: MementoMemory, top_k: int = 3):
        scores = []
        for i, case in enumerate(memory.cases):
            s = self.score(q_intent, q_hist, q_date, case)
            scores.append((s.item(), i))
        scores.sort(reverse=True)
        return [i for _, i in scores[: top_k]]

    def update(self, logits: torch.Tensor, reward: float):
        loss = -reward * logits.mean()
        loss.backward()


class MementoAgent:
    def __init__(self, vec: DataVectorizer, model: VIMN_Lite, head: IntentContrastiveHead, memory: MementoMemory, policy: MementoPolicy):
        self.vec = vec
        self.model = model
        self.head = head
        self.memory = memory
        self.policy = policy
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def encode_query(self, traj: str):
        poi, act, time = self.vec.vectorize_sequence([traj])
        with torch.no_grad():
            z = self.model(poi, act, time)
            z = self.head(z)
            z = nn.functional.normalize(z, dim=-1)
        items = self.vec._parse_route(traj)
        hours = [h for _, _, h in items]
        hist = torch.zeros(24, dtype=torch.float)
        for h in hours:
            hist[h % 24] += 1.0
        if hist.sum() > 0:
            hist = hist / hist.sum()
        date_str = traj.split(": ")[0].split(" ")[-1]
        return z.squeeze(0), hist, date_str

    def step(self, query_traj: str, top_k: int = 3):
        q_intent, q_hist, q_date = self.encode_query(query_traj)
        idxs = self.policy.select(q_intent, q_hist, q_date, self.memory, top_k)
        return idxs

    def reward_from_alignment(self, query_traj: str, case: MementoCase):
        q_items = self.vec._parse_route(query_traj)
        q_acts = set([a for _, a, _ in q_items])
        inter = len(q_acts.intersection(case.act_set))
        union = len(q_acts.union(case.act_set))
        j = 0.0 if union == 0 else inter / union
        q_hist = torch.zeros(24, dtype=torch.float)
        for _, _, h in q_items:
            q_hist[h % 24] += 1.0
        if q_hist.sum() > 0:
            q_hist = q_hist / q_hist.sum()
        t = torch.dot(q_hist, case.hour_hist).item()
        return 0.7 * j + 0.3 * t

    def update(self, query_traj: str, sel_indices):
        q_intent, q_hist, q_date = self.encode_query(query_traj)
        logits = []
        rewards = []
        for i in sel_indices:
            s = self.policy.score(q_intent, q_hist, q_date, self.memory.cases[i])
            r = self.reward_from_alignment(query_traj, self.memory.cases[i])
            logits.append(s.view(1))
            rewards.append(r)
            self.memory.update_reward(i, r)
        if len(logits) == 0:
            return
        logits = torch.cat(logits, dim=0)
        reward = sum(rewards) / len(rewards)
        self.opt.zero_grad()
        self.policy.update(logits, reward)
        self.opt.step()