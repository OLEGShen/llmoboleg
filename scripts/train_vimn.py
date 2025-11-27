import argparse
import os
import sys
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda import amp

sys.path.append(os.path.abspath('.'))
from engine.experimental.vimn import DataVectorizer, VIMN_Lite, IntentContrastiveHead
from engine.experimental.datasets import TrajectoryPairDataset


def vectorize_pair(vec: DataVectorizer, anchor: str, positive: str, negs):
    poi_a, act_a, time_a = vec.vectorize_sequence([anchor])
    poi_p, act_p, time_p = vec.vectorize_sequence([positive])
    poi_n, act_n, time_n = vec.vectorize_sequence(negs)
    return (poi_a, act_a, time_a), (poi_p, act_p, time_p), (poi_n, act_n, time_n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=str, default='2019')
    parser.add_argument('--id', type=str, default='934')
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--num_neg', type=int, default=8)
    parser.add_argument('--precision', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16'])
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

    vec = DataVectorizer('./data/loc_map.pkl', './data/location_activity_map.pkl',
                         allowed_poi_ids=agg_allowed_poi, allowed_act_names=agg_allowed_act)
    model = VIMN_Lite(num_pois=len(vec.poi_vocab), num_acts=len(vec.act_vocab),
                      embed_dim=args.embed_dim, hidden_dim=args.hidden_dim)
    head = IntentContrastiveHead(args.hidden_dim)

    use_cuda = torch.cuda.is_available() if args.device in ['auto', 'cuda'] else False
    device = torch.device('cuda' if use_cuda else 'cpu')
    if args.precision == 'fp16':
        amp_dtype = torch.float16
    elif args.precision == 'bf16':
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32
    try:
        model = model.to(device=device, dtype=(amp_dtype if device.type == 'cuda' else torch.float32))
        head = head.to(device=device, dtype=(amp_dtype if device.type == 'cuda' else torch.float32))
    except RuntimeError as e:
        if 'out of memory' in str(e).lower() and device.type == 'cuda':
            torch.cuda.empty_cache()
            print('CUDA OOM, falling back to CPU')
            device = torch.device('cpu')
            amp_dtype = torch.float32
            model = model.to(device)
            head = head.to(device)
        else:
            raise

    dataset = TrajectoryPairDataset(f'./data/{args.year}/{args.id}.pkl', num_neg=args.num_neg)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=False)

    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    scaler = amp.GradScaler(enabled=(device.type == 'cuda' and amp_dtype == torch.float16))

    def encode_one(poi, act, time):
        with torch.set_grad_enabled(True):
            with amp.autocast(enabled=(device.type == 'cuda' and amp_dtype != torch.float32), dtype=amp_dtype if device.type == 'cuda' else None):
                z = model(poi.to(device), act.to(device), time.to(device))
                z = head(z)
            z = nn.functional.normalize(z, dim=-1)
        return z

    model.train()
    for epoch in range(args.epochs):
        for anchor, positive, negs in loader:
            opt.zero_grad(set_to_none=True)
            poi_a, act_a, time_a = vec.vectorize_sequence(list(anchor))
            poi_p, act_p, time_p = vec.vectorize_sequence(list(positive))
            z_a = encode_one(poi_a, act_a, time_a)
            z_p = encode_one(poi_p, act_p, time_p)
            logits = z_a @ z_p.T
            labels = torch.arange(logits.size(0), device=device)
            with amp.autocast(enabled=(device.type == 'cuda' and amp_dtype != torch.float32), dtype=amp_dtype if device.type == 'cuda' else None):
                loss = loss_fn(logits, labels)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
        print(f'Epoch {epoch+1}/{args.epochs} done')

    ckpt_dir = './engine/experimental/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    suffix = f"y{args.year}_id_{args.id}_ed{args.embed_dim}_hd{args.hidden_dim}_b{args.batch}_ep{args.epochs}_prec{args.precision}"
    path_default = f"{ckpt_dir}/vimn_lite.pt"
    path_param = f"{ckpt_dir}/vimn_lite_{suffix}.pt"
    torch.save({'model': model.state_dict(), 'head': head.state_dict()}, path_default)
    torch.save({'model': model.state_dict(), 'head': head.state_dict()}, path_param)
    print(f'Saved to {path_default} and {path_param}')


if __name__ == '__main__':
    main()