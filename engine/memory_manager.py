import torch
from typing import Optional, Dict


class DualMemory:
    def __init__(self, vimn_model, vec, act_vocab: Dict[str, int]):
        self.vimn = vimn_model
        self.vec = vec
        self.h: Optional[torch.Tensor] = None
        self.act_vocab = act_vocab

    def reset(self):
        self.h = None

    def ingest_location(self, poi_id: int, act_name: str = None, time_hour: int = 0):
        if act_name is None:
            act_name = self.vec.id2act.get(int(poi_id), 'UNK')
        act_idx = self.vec.act_vocab.get(act_name, self.vec.act_vocab.get('UNK'))
        poi_idx = self.vec.poi_vocab.get(int(poi_id), self.vec.poi_vocab.get('UNK'))
        self.h = self.vimn.step(poi_idx, act_idx, int(time_hour) % 24, self.h)
        return self.h

    def get_state(self):
        return self.h