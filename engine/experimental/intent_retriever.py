import torch
import torch.nn as nn
from .vimn import VIMN_Lite, DataVectorizer, IntentContrastiveHead


class IntentRetriever:
    def __init__(self, vec: DataVectorizer, model: VIMN_Lite, head: IntentContrastiveHead,
                 train_trajs, top_k=6, test_trajs=None):
        self.vec = vec
        self.model = model
        self.head = head
        self.train_trajs = train_trajs
        self.top_k = top_k
        self.nodes = self.train_trajs
        self.train_intents = self._encode_all(train_trajs)
        self.test_trajs = test_trajs or []
        self.test_map = {}
        for t in self.test_trajs:
            try:
                d = t.split(": ")[0].split(" ")[-1]
                self.test_map[d] = t
            except Exception:
                pass

    def _encode_all(self, trajs):
        intents = []
        for t in trajs:
            try:
                poi, act, time = self.vec.vectorize_sequence([t])
                with torch.no_grad():
                    z = self.model(poi, act, time)
                    z = self.head(z)
                    z = nn.functional.normalize(z, dim=-1)
                intents.append(z.squeeze(0))
            except Exception:
                intents.append(None)
        return intents

    def retrieve(self, query):
        q = query
        if isinstance(query, str) and len(query) == 10 and query[4] == '-' and query[7] == '-':
            q = self.test_map.get(query, None)
        if q is None:
            return []
        poi, act, time = self.vec.vectorize_sequence([q])
        with torch.no_grad():
            zq = self.model(poi, act, time)
            zq = self.head(zq)
            zq = nn.functional.normalize(zq, dim=-1).squeeze(0)
        scores = []
        for idx, zi in enumerate(self.train_intents):
            if zi is None:
                continue
            sim = torch.dot(zq, zi)
            scores.append((sim.item(), idx))
        scores.sort(reverse=True)
        indices = [j for _, j in scores[: self.top_k]]
        return [self.train_trajs[j] for j in indices]