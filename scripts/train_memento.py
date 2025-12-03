import os
import sys
import argparse
import pickle
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import random
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath('.'))
from engine.memento_policy import MementoTrainer, construct_training_samples
from engine.utilities.retrieval_helper import map_traj2mat


def parse_date_from_route(route: str) -> str:
    return route.split(": ")[0].split(" ")[-1]


def parse_first_step(route: str) -> Tuple[str, int]:
    part = route.split(": ", 1)[1]
    seg = part.split(',')[0].strip()
    import re
    m_loc = re.search(r"(.+?)#(\d+)", seg)
    m_tm = re.search(r"(\d{2}):(\d{2})", seg)
    if not m_loc or not m_tm:
        # Fallback: try legacy split
        toks = part.replace(",", " at ").split(" at ")
        loc = toks[0].strip()
        poi_id = int(loc.split("#")[-1])
        loc_name = loc.split("#")[0].strip()
        return loc_name, poi_id
    loc_name = m_loc.group(1).strip()
    poi_id = int(m_loc.group(2))
    return loc_name, poi_id


def extract_items(route: str) -> List[Tuple[str, int, int]]:
    part = route.split(": ", 1)[1]
    segs = [s.strip() for s in part.split(',')]
    items: List[Tuple[str, int, int]] = []
    import re
    for seg in segs:
        m_loc = re.search(r"(.+?)#(\d+)", seg)
        m_tm = re.search(r"(\d{2}):(\d{2})", seg)
        if m_loc and m_tm:
            loc_name = m_loc.group(1).strip()
            pid = int(m_loc.group(2))
            hh = int(m_tm.group(1)); mm = int(m_tm.group(2))
            items.append((loc_name, pid, hh*60+mm))
    return items


def build_act_map(loc_cat: Dict[str, str]) -> Dict[str, int]:
    acts = sorted(set(loc_cat.values()))
    return {a: i for i, a in enumerate(acts)}


def route_to_vec(route: str, loc_cat: Dict[str, str], act_map: Dict[str, int], interval: int = 60) -> torch.Tensor:
    bins = int(1440 / interval)
    mat = torch.zeros((bins, len(act_map)), dtype=torch.float32)
    items = extract_items(route)
    for loc_name, _, tmin in items:
        cat = loc_cat.get(loc_name)
        if cat is None or cat not in act_map:
            continue
        b = max(0, min(bins-1, tmin // interval))
        mat[b, act_map[cat]] += 1.0
    return mat.reshape(-1)


class SampleDataset(Dataset):
    def __init__(self):
        self.state_list: List[torch.Tensor] = []
        self.mem_list: List[torch.Tensor] = []
        self.lab_list: List[torch.Tensor] = []
    def append(self, s: torch.Tensor, m: torch.Tensor, l: torch.Tensor):
        self.state_list.append(s)
        self.mem_list.append(m)
        self.lab_list.append(l)
    def __len__(self):
        return len(self.lab_list)
    def __getitem__(self, idx):
        return self.state_list[idx], self.mem_list[idx], self.lab_list[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='2019')
    parser.add_argument('--id', type=int, default=None)
    parser.add_argument('--ids', type=str, default=None, help='逗号分隔的 id 列表或 all')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    folder = f"./data/{args.dataset}/"

    def load_id_list() -> List[int]:
        if args.ids is None and args.id is not None:
            return [args.id]
        if args.ids is None:
            files = [fn for fn in os.listdir(folder) if fn.endswith('.pkl')]
            out = []
            for fn in files:
                try:
                    out.append(int(os.path.splitext(fn)[0]))
                except Exception:
                    pass
            return sorted(out)
        if args.ids.strip().lower() == 'all':
            files = [fn for fn in os.listdir(folder) if fn.endswith('.pkl')]
            out = []
            for fn in files:
                try:
                    out.append(int(os.path.splitext(fn)[0]))
                except Exception:
                    pass
            return sorted(out)
        out = []
        for tok in args.ids.split(','):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(int(tok))
            except Exception:
                pass
        return sorted(out)

    id_list = load_id_list()
    random.shuffle(id_list)

    # 统一活动类别空间，保证不同 id 的向量维度一致
    act_set: set = set()
    for id_ in id_list:
        with open(os.path.join(folder, f"{id_}.pkl"), "rb") as f:
            att = pickle.load(f)
        loc_cat_i = att[11]
        for loc, cat in loc_cat_i.items():
            act_set.add(cat)
    act_map_global: Dict[str, int] = {a: i for i, a in enumerate(sorted(act_set))}

    # 收集所有训练样本
    dataset = SampleDataset()
    for id_ in id_list:
        with open(os.path.join(folder, f"{id_}.pkl"), "rb") as f:
            att = pickle.load(f)
        train_routine_list, loc_cat = att[0], att[11]
        for route in train_routine_list:
            # 当前状态向量
            v_route = route_to_vec(route, loc_cat, act_map_global)
            # 候选：同 id 内基于余弦相似度的 top_k
            scored: List[Tuple[float, str]] = []
            for cand in train_routine_list:
                if cand == route:
                    continue
                v_cand = route_to_vec(cand, loc_cat, act_map_global)
                if v_cand.numel() == 0 or v_route.numel() == 0:
                    continue
                sim = float(F.cosine_similarity(v_route.unsqueeze(0), v_cand.unsqueeze(0)).item())
                scored.append((sim, cand))
            scored.sort(key=lambda x: x[0], reverse=True)
            candidates = [r for _, r in scored[:args.top_k]]
            # GT 标签
            loc_name, poi_id = parse_first_step(route)
            gt_act = loc_cat.get(loc_name, None)
            # 构造样本
            cand_list: List[dict] = []
            for cand in candidates:
                c_loc_name, c_poi_id = parse_first_step(cand)
                c_act = loc_cat.get(c_loc_name, None)
                emb = route_to_vec(cand, loc_cat, act_map_global)
                cand_list.append({'memory_emb': emb, 'act_name': c_act, 'poi_id': c_poi_id})
            mem_embs, labels = construct_training_samples(gt_act, poi_id, cand_list)
            if mem_embs.numel() == 0:
                continue
            state_embs = v_route.unsqueeze(0).expand(mem_embs.shape[0], -1)
            for i in range(mem_embs.shape[0]):
                dataset.append(state_embs[i], mem_embs[i], labels[i])

    state_dim = int(1440 / 60) * len(act_map_global)
    input_dim = state_dim * 2
    trainer = MementoTrainer(input_dim=input_dim, hidden_dim=1024, lr=args.lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer.model.to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        trainer.model = torch.nn.DataParallel(trainer.model)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    for epoch in range(args.epochs):
        total_loss = 0.0
        steps = 0
        for state_batch, mem_batch, lab_batch in loader:
            loss, _ = trainer.train_batch(state_batch.to(device), mem_batch.to(device), lab_batch.to(device))
            total_loss += loss
            steps += 1
        avg = total_loss / max(steps, 1)
        print(f"epoch={epoch+1}/{args.epochs} avg_loss={avg:.4f} steps={steps}")

    os.makedirs('./engine/experimental/checkpoints', exist_ok=True)
    save_path = './engine/experimental/checkpoints/memento_policy.pt'
    sd = trainer.model.state_dict() if not isinstance(trainer.model, torch.nn.DataParallel) else trainer.model.module.state_dict()
    torch.save({'state_dict': sd, 'input_dim': input_dim, 'hidden_dim': 1024}, save_path)
    print(f"saved: {save_path}")


if __name__ == '__main__':
    main()