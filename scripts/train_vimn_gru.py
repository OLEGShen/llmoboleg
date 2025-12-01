import argparse
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from engine.vimn_core import VIMN
from engine.experimental.vimn import DataVectorizer


class NextActDataset(Dataset):
    def __init__(self, pkl_path: str, allowed_poi_ids=None, allowed_act_names=None, min_len: int = 2):
        with open(pkl_path, 'rb') as f:
            att = pickle.load(f)
        self.train = att[0]
        self.allowed_poi_ids = allowed_poi_ids
        self.allowed_act_names = allowed_act_names
        self.vec = DataVectorizer('./data/loc_map.pkl', './data/location_activity_map.pkl',
                                  allowed_poi_ids=allowed_poi_ids,
                                  allowed_act_names=allowed_act_names)
        self.samples = []
        for traj in self.train:
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
                    self.samples.append((x_poi.squeeze(0), x_act.squeeze(0), x_tim.squeeze(0), y_act.squeeze(0)))
            except Exception:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, a, t, y = self.samples[idx]
        return p, a, t, y


def collate_pad(batch):
    max_len = max(x[0].size(0) for x in batch)
    B = len(batch)
    def pad1d(x):
        pad = torch.zeros(max_len - x.size(0), dtype=torch.long)
        return torch.cat([x, pad], dim=0)
    poi = torch.stack([pad1d(p) for p,_,_,_ in batch], dim=0)
    act = torch.stack([pad1d(a) for _,a,_,_ in batch], dim=0)
    tim = torch.stack([pad1d(t) for _,_,t,_ in batch], dim=0)
    y = torch.stack([y for *_, y in batch], dim=0)
    return poi, act, tim, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=str, default='2019')
    parser.add_argument('--id', type=str, default='934')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()

    agg_allowed_poi = None
    agg_allowed_act = None
    agg_path = f'./data/{args.year}/{args.id}.pkl'
    try:
        with open(agg_path, 'rb') as f:
            att = pickle.load(f)
        if isinstance(att, (list, tuple)) and len(att) >= 4:
            agg_allowed_poi = att[2] if att[2] else None
            agg_allowed_act = att[3] if att[3] else None
    except Exception:
        pass

    dataset = NextActDataset(agg_path, allowed_poi_ids=agg_allowed_poi, allowed_act_names=agg_allowed_act)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=False, collate_fn=collate_pad)

    num_pois = len(dataset.vec.poi_vocab)
    num_acts = len(dataset.vec.act_vocab)
    model = VIMN(num_pois=num_pois, num_acts=num_acts, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim)

    use_cuda = torch.cuda.is_available() if args.device in ['auto', 'cuda'] else False
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        for poi, act, tim, y in loader:
            poi, act, tim, y = poi.to(device), act.to(device), tim.to(device), y.to(device)
            opt.zero_grad()
            h = model(poi, act, tim)
            logits = model.translate(h)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
        print(f'Epoch {epoch+1}/{args.epochs} done')

    ckpt_dir = './engine/experimental/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    path = f"{ckpt_dir}/vimn_best_gru.pt"
    torch.save({'model': model.state_dict(), 'meta': {
        'embed_dim': args.embed_dim,
        'hidden_dim': args.hidden_dim,
        'allowed_poi_ids': agg_allowed_poi,
        'allowed_act_names': agg_allowed_act,
        'num_pois': num_pois,
        'num_acts': num_acts,
    }}, path)
    print(f'Saved GRU VIMN to {path}')


if __name__ == '__main__':
    main()