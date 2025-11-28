import torch
import torch.nn as nn
from typing import List


class IntentionTranslator:
    def __init__(self, act_id2name: List[str]):
        self.act_id2name = act_id2name

    def get_natural_language_hint(self, logits: torch.Tensor, top_k: int = 3) -> str:
        probs = torch.softmax(logits.squeeze(0), dim=-1)
        k = min(top_k, probs.shape[-1])
        topv, topi = torch.topk(probs, k)
        cats = []
        for idx in topi.tolist():
            try:
                cats.append(str(self.act_id2name[idx]))
            except Exception:
                cats.append('Unknown')
        text = f"我的习惯记忆通常在此时会去：[{', '.join(cats)}]。"
        return text