import os
import json
import pickle
import sys
import argparse
import time
sys.path.append(os.path.abspath('.'))

from scripts.train_vimn_gru import NextActDataset
from engine.vimn_core import VIMN
from engine.experimental.vimn import DataVectorizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one(year: str, pid: str, epochs: int, batch: int, embed_dim: int, hidden_dim: int, device: str):
    agg_path = f'./data/{year}/{pid}.pkl'
    with open(agg_path, 'rb') as f:
        att = pickle.load(f)
    allowed_poi = att[2] if len(att) >= 3 and att[2] else None
    allowed_act = att[3] if len(att) >= 4 and att[3] else None
    dataset = NextActDataset(agg_path, allowed_poi_ids=allowed_poi, allowed_act_names=allowed_act)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=False,
                        collate_fn=lambda b: __import__('scripts.train_vimn_gru', fromlist=['collate_pad']).collate_pad(b))
    vec = dataset.vec
    model = VIMN(len(vec.poi_vocab), len(vec.act_vocab), embed_dim, hidden_dim)
    use_cuda = torch.cuda.is_available() if device in ['auto', 'cuda'] else False
    dev = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    model.train()
    for ep in range(epochs):
        for poi, act, tim, y in loader:
            poi, act, tim, y = poi.to(dev), act.to(dev), tim.to(dev), y.to(dev)
            opt.zero_grad()
            h = model(poi, act, tim)
            logits = model.translate(h)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
    dur = time.time() - t0
    ckpt_dir = './engine/experimental/checkpoints/batch'
    os.makedirs(ckpt_dir, exist_ok=True)
    path = f"{ckpt_dir}/vimn_best_gru_{year}_{pid}.pt"
    torch.save({'model': model.state_dict(), 'meta': {
        'embed_dim': embed_dim,
        'hidden_dim': hidden_dim,
        'allowed_poi_ids': allowed_poi,
        'allowed_act_names': allowed_act,
        'num_pois': len(vec.poi_vocab),
        'num_acts': len(vec.act_vocab),
        'year': year,
        'pid': pid,
        'train_samples': len(dataset),
        'duration_sec': dur,
    }}, path)
    return {'path': path, 'duration_sec': dur, 'samples': len(dataset)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=str, default='2019')
    parser.add_argument('--ids', type=str, nargs='*', default=['934','835','4105','4396','2513'])
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    results = {}
    for pid in args.ids:
        try:
            info = train_one(args.year, str(pid), args.epochs, args.batch, args.embed_dim, args.hidden_dim, args.device)
            results[str(pid)] = info
            print(f"Trained {pid}: {info}")
        except Exception as e:
            results[str(pid)] = {'error': str(e)}
            print(f"Failed {pid}: {e}")
    os.makedirs('./engine/experimental/checkpoints/batch', exist_ok=True)
    with open('./engine/experimental/checkpoints/batch/train_summary.json','w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('Summary saved to checkpoints/batch/train_summary.json')


if __name__ == '__main__':
    main()