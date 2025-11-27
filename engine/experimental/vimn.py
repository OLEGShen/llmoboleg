import torch
import torch.nn as nn
import pickle
import re
from typing import List, Tuple


class DataVectorizer:
    def __init__(self, loc_map_path: str, act_map_path: str,
                 allowed_poi_ids=None, allowed_act_names=None):
        with open(loc_map_path, 'rb') as f:
            self.loc_map = pickle.load(f)
        with open(act_map_path, 'rb') as f:
            self.act_map = pickle.load(f)

        poi_ids = []
        for v in self.loc_map.values():
            try:
                id_part = int(str(v).split('#')[-1])
                poi_ids.append(id_part)
            except Exception:
                continue
        base_poi = sorted(set(poi_ids))
        if allowed_poi_ids is not None:
            allow_set = set(int(x) for x in allowed_poi_ids)
            base_poi = [pid for pid in base_poi if pid in allow_set]
        self.poi_vocab = {pid: i for i, pid in enumerate(base_poi)}
        if 'UNK' not in self.poi_vocab:
            self.poi_vocab['UNK'] = len(self.poi_vocab)

        self.id2act = self._align_poi_activities()
        if allowed_poi_ids is not None:
            act_from_allowed_poi = sorted(set(self.id2act.get(pid) for pid in base_poi if self.id2act.get(pid) is not None))
            base_act = act_from_allowed_poi
        else:
            base_act = sorted(set(self.act_map.values()))
        if allowed_act_names is not None:
            allow_act = set(str(x) for x in allowed_act_names)
            base_act = [a for a in base_act if str(a) in allow_act]
        self.act_vocab = {v: i for i, v in enumerate(base_act)}
        if 'UNK' not in self.act_vocab:
            self.act_vocab['UNK'] = len(self.act_vocab)

    def _align_poi_activities(self):
        """建立 POI ID 到 Activity 的映射。act_map 的 key 是地点名，需按 loc_map 的格式拼接。
        loc_map 的 key 形如: "Location Name (lat, lon)"，value 是唯一 ID。
        我们将 act_map 的 key 与 loc_map 的 key 前半部分匹配，得到对应的 Activity。
        """
        id2act = {}
        # 预构造 name -> activity 的快速查找
        name2act = {}
        for name, act in self.act_map.items():
            name2act[str(name).strip()] = act

        # 提取 loc_map 中的名称部分
        pattern = r"^(.*?) \(.*\)$"
        for key, poi_id in self.loc_map.items():
            m = re.match(pattern, str(key))
            if not m:
                continue
            name = str(m.group(1)).strip()
            act = name2act.get(name)
            try:
                id_part = int(str(self.loc_map[key]).split('#')[-1])
            except Exception:
                id_part = None
            if act is not None and id_part is not None:
                id2act[id_part] = act
        return id2act

    def _parse_route(self, route: str) -> List[Tuple[int, int, int]]:
        """将单条轨迹字符串解析为 (poi_id_idx, act_id_idx, hour) 列表。
        轨迹格式示例: "Activities at 2019-01-02: Cafe#123 at 08:30, Library#456 at 11:00"
        """
        # 提取 "地点#ID at HH:MM" 模式
        items = []
        parts = route.split(": ")[-1]
        tokens = [t.strip() for t in parts.split(",")]
        for tok in tokens:
            m = re.match(r"^(.*?)#(\d+)\s+at\s+(\d{2}):(\d{2})(?::\d{2})?", tok)
            if not m:
                m2 = re.match(r"^(.*?)\s+at\s+(\d{2}):(\d{2})(?::\d{2})?", tok)
                if not m2:
                    continue
                hour = int(m2.group(2))
                poi_idx = self.poi_vocab.get('UNK')
                act_idx = self.act_vocab.get('UNK')
                items.append((poi_idx, act_idx, hour))
                continue
            poi_id = int(m.group(2))
            hour = int(m.group(3))
            poi_idx = self.poi_vocab.get(poi_id, self.poi_vocab.get('UNK'))
            act_name = self.id2act.get(poi_id)
            act_idx = self.act_vocab.get(act_name, self.act_vocab.get('UNK'))
            items.append((poi_idx, act_idx, hour))
        return items

    def vectorize_sequence(self, trajectory_list: List[str]):
        """将轨迹列表转换为 (Batch, Seq_Len) 的张量。返回: poi_ids, act_ids, time_ids"""
        batch_items = [self._parse_route(traj) for traj in trajectory_list]
        # 过滤空样本
        batch_items = [items for items in batch_items if len(items) > 0]
        if len(batch_items) == 0:
            raise ValueError("No valid trajectory items parsed")

        max_len = max(len(items) for items in batch_items)
        B = len(batch_items)

        poi_ids = torch.zeros((B, max_len), dtype=torch.long)
        act_ids = torch.zeros((B, max_len), dtype=torch.long)
        time_ids = torch.zeros((B, max_len), dtype=torch.long)

        for i, items in enumerate(batch_items):
            for j, (p_idx, a_idx, h) in enumerate(items):
                poi_ids[i, j] = p_idx
                act_ids[i, j] = a_idx
                time_ids[i, j] = h % 24

        return poi_ids, act_ids, time_ids


class VIMN_Lite(nn.Module):
    """
    轻量级意图编码器，融合 Mobility-LLM 的思想
    """
    def __init__(self, num_pois: int, num_acts: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.poi_emb = nn.Embedding(num_pois, embed_dim)
        self.act_emb = nn.Embedding(num_acts, embed_dim)
        self.time_emb = nn.Embedding(24, embed_dim)

        self.input_proj = nn.Linear(embed_dim * 3, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.intent_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, poi_ids: torch.Tensor, act_ids: torch.Tensor, time_ids: torch.Tensor) -> torch.Tensor:
        p = self.poi_emb(poi_ids)
        a = self.act_emb(act_ids)
        t = self.time_emb(time_ids)
        x = torch.cat([p, a, t], dim=-1)
        x = self.input_proj(x)
        encoded = self.encoder(x)
        intent_vector = self.intent_head(encoded[:, -1, :])
        return intent_vector


class IntentContrastiveHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)