import os
import pickle
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from engine.vimn_core import VIMN
from engine.experimental.vimn import DataVectorizer


class GlobalNextActDataset(Dataset):
    def __init__(self, year: str, ids, min_len: int = 2, min_poi_freq: int = 2):
        self.records = []
        self.user2idx = {}
        for uid in ids:
            pkl = f'./data/{year}/{uid}.pkl'
            try:
                with open(pkl, 'rb') as f:
                    att = pickle.load(f)
                train = att[0]
                self.records.append((str(uid), train))
                self.user2idx[str(uid)] = len(self.user2idx)
            except Exception:
                continue
        used_poi = {}
        for _, trajs in self.records:
            for route in trajs:
                try:
                    parts = route.split(': ')[-1]
                    toks = [t.strip() for t in parts.split(',')]
                    for tok in toks:
                        m = __import__('re').match(r"^(.*?)#(\d+)\s+at\s+(\d{2}):(\d{2})", tok)
                        if m:
                            pid = int(m.group(2))
                            used_poi[pid] = used_poi.get(pid, 0) + 1
                except Exception:
                    continue
        allowed_poi_ids = [pid for pid, c in used_poi.items() if c >= min_poi_freq]
        base_vec = DataVectorizer('./data/loc_map.pkl', './data/location_activity_map.pkl')
        allowed_act_names = sorted(set(base_vec.id2act.get(pid) for pid in allowed_poi_ids if base_vec.id2act.get(pid) is not None))
        self.allowed_poi_ids = allowed_poi_ids
        self.allowed_act_names = allowed_act_names
        self.vec = DataVectorizer('./data/loc_map.pkl', './data/location_activity_map.pkl', allowed_poi_ids=allowed_poi_ids, allowed_act_names=allowed_act_names)
        self.samples = []
        self.act_counts = {}
        for uid, trajs in self.records:
            for traj in trajs:
                try:
                    poi_ids, act_ids, time_ids = self.vec.vectorize_sequence([traj])
                    L = poi_ids.size(1)
                    if L < min_len:
                        continue
                    for t in range(1, L):
                        x_poi = poi_ids[:, :t]
                        x_act = act_ids[:, :t]
                        x_tim = time_ids[:, :t]
                        y_act = act_ids[:, t]
                        user_idx = self.user2idx[uid]
                        y_idx = int(y_act.squeeze(0).item())
                        self.act_counts[y_idx] = self.act_counts.get(y_idx, 0) + 1
                        self.samples.append((x_poi.squeeze(0), x_act.squeeze(0), x_tim.squeeze(0), y_act.squeeze(0), user_idx))
                except Exception:
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, a, t, y, u = self.samples[idx]
        return p, a, t, y, u

    def make_class_weights(self):
        import torch as _t
        total = sum(self.act_counts.values()) or 1
        w = _t.ones(len(self.vec.act_vocab), dtype=_t.float)
        for i, c in self.act_counts.items():
            w[i] = total / max(1, c)
        w = w / w.mean()
        return w


def collate_pad_user(batch):
    max_len = max(x[0].size(0) for x in batch)
    def pad1d(x):
        pad = torch.zeros(max_len - x.size(0), dtype=torch.long)
        return torch.cat([x, pad], dim=0)
    poi = torch.stack([pad1d(p) for p,_,_,_,_ in batch], dim=0)
    act = torch.stack([pad1d(a) for _,a,_,_,_ in batch], dim=0)
    tim = torch.stack([pad1d(t) for _,_,t,_,_ in batch], dim=0)
    y = torch.stack([y for *_, y,_ in batch], dim=0)
    u = torch.tensor([u for *_, u in batch], dtype=torch.long)
    return poi, act, tim, y, u


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=str, default='2019')
    parser.add_argument('--ids', type=str, nargs='*', default=['934','835','4105','4396','2513'])
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--user_embed_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--min_poi_freq', type=int, default=2)
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()

    dataset = GlobalNextActDataset(args.year, args.ids, min_len=2, min_poi_freq=args.min_poi_freq)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=False, collate_fn=collate_pad_user)
    num_users = len(dataset.user2idx)
    num_pois = len(dataset.vec.poi_vocab)
    num_acts = len(dataset.vec.act_vocab)
    model = VIMN(num_pois=num_pois, num_acts=num_acts, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
                 num_users=num_users, user_embed_dim=args.user_embed_dim, dropout=args.dropout)

    use_cuda = torch.cuda.is_available() if args.device in ['auto', 'cuda'] else False
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    class_w = dataset.make_class_weights().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and args.amp))
    model.train()
    for ep in range(args.epochs):
        for poi, act, tim, y, u in loader:
            poi, act, tim, y, u = poi.to(device), act.to(device), tim.to(device), y.to(device), u.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda' and args.amp)):
                h = model(poi, act, tim, user_ids=u)
                logits = model.translate(h)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        print(f'Epoch {ep+1}/{args.epochs} done')

    ckpt_dir = './engine/experimental/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    path = f"{ckpt_dir}/vimn_global_gru_{args.year}_train_ids.pt"
    torch.save({'model': model.state_dict(), 'meta': {
        'embed_dim': args.embed_dim,
        'hidden_dim': args.hidden_dim,
        'user_embed_dim': args.user_embed_dim,
        'num_users': num_users,
        'num_pois': num_pois,
        'num_acts': num_acts,
        'user2idx': dataset.user2idx,
        'ids': args.ids,
        'year': args.year,
        'allowed_poi_ids': dataset.allowed_poi_ids,
        'allowed_act_names': dataset.allowed_act_names,
    }}, path)
    print(f'Saved global GRU VIMN to {path}')


if __name__ == '__main__':
    main()