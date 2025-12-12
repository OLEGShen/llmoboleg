import math
import os
from typing import List, Tuple, Dict, Optional

import torch
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def _haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


WEEKMAP = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}


def _to_text(weekday: int, time_str: str, poi_cat: str) -> str:
    w = WEEKMAP.get(int(weekday) % 7, '周一')
    return f"{w} {time_str} 访问了 [{poi_cat}]"


class SemanticMemento:
    def __init__(self,
                 model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 dist_guard_km: float = 20.0):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if model_path is not None:
            mp = model_path
        else:
            try:
                here = os.path.dirname(os.path.abspath(__file__))
                local_models = os.path.abspath(os.path.join(here, '..', '..', 'models'))
                local_minilm = os.path.join(local_models, 'all-MiniLM-L6-v2')
                if os.path.isdir(local_minilm) and (
                    os.path.isfile(os.path.join(local_minilm, 'modules.json')) or
                    os.path.isfile(os.path.join(local_minilm, 'config.json'))
                ):
                    mp = local_minilm
                else:
                    mp = 'sentence-transformers/all-MiniLM-L6-v2'
            except Exception:
                mp = 'sentence-transformers/all-MiniLM-L6-v2'
        if SentenceTransformer is None:
            try:
                print('MiniLM: sentence-transformers not installed')
            except Exception:
                pass
            raise RuntimeError('sentence-transformers not installed')
        try:
            self.encoder = SentenceTransformer(mp, device=self.device)
        except Exception:
            self.device = 'cpu'
            self.encoder = SentenceTransformer(mp, device=self.device)
        try:
            print(f"MiniLM: encoder init path={mp}, device={self.device}")
        except Exception:
            pass
        self.encoder = self.encoder
        self.dist_guard_km = float(dist_guard_km)
        self.mem_texts: List[str] = []
        self.mem_vecs: Optional[torch.Tensor] = None
        self.mem_meta: List[Dict] = []

    def build_memory(self, records: List[Dict]):
        texts = []
        metas = []
        for r in records:
            txt = _to_text(r.get('weekday', 0), r.get('time', '00:00'), r.get('poi_cat', '未知'))
            texts.append(txt)
            metas.append({
                'poi_id': r.get('poi_id'),
                'lat': r.get('lat'),
                'lng': r.get('lng'),
                'weekday': r.get('weekday'),
                'time': r.get('time'),
                'raw': r,
                'route': r.get('route')
            })
        emb = self.encoder.encode(texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True)
        self.mem_texts = texts
        self.mem_vecs = emb.detach().cpu()
        self.mem_meta = metas

    def retrieve(self,
                 query_text: str,
                 curr_lat: float,
                 curr_lng: float,
                 top_k: int = 5) -> Tuple[List[Dict], float]:
        if self.mem_vecs is None or len(self.mem_meta) == 0:
            return [], 0.0
        q = self.encoder.encode([query_text], batch_size=1, convert_to_tensor=True, normalize_embeddings=True)
        q = q.detach().cpu()
        sims = torch.matmul(self.mem_vecs, q.squeeze(0))
        vals, idxs = torch.topk(sims, k=min(top_k, sims.shape[0]))
        cand = []
        for s, i in zip(vals.tolist(), idxs.tolist()):
            m = self.mem_meta[i]
            d = _haversine_km(curr_lng, curr_lat, m.get('lng', 0.0), m.get('lat', 0.0))
            cand.append({
                'score': float(s),
                'distance_km': float(d),
                'meta': m
            })
        filtered = [c for c in cand if c['distance_km'] <= self.dist_guard_km]
        if len(filtered) == 0:
            filtered = cand
        top1_score = float(filtered[0]['score']) if len(filtered) > 0 else 0.0
        return filtered, top1_score