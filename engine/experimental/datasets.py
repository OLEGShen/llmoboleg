import random
import pickle
from typing import List, Tuple

class TrajectoryPairDataset:
    def __init__(self, pkl_path: str, min_len: int = 1, num_neg: int = 4):
        with open(pkl_path, 'rb') as f:
            att = pickle.load(f)
        self.train = att[0] if isinstance(att[0], (list, tuple)) else []
        self.num_neg = num_neg
        self.indices = list(range(len(self.train)))
        self.pairs = []
        for i in range(1, len(self.train)):
            a = self.train[i]
            p = self.train[i - 1]
            if len(a) >= min_len and len(p) >= min_len:
                self.pairs.append((i, i - 1))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        anchor = self.train[i]
        positive = self.train[j]
        negs = []
        pool = [k for k in self.indices if k != i and k != j]
        random.shuffle(pool)
        for k in pool[: self.num_neg]:
            negs.append(self.train[k])
        return anchor, positive, negs