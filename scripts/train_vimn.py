import argparse
import os
import sys
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()

    vec = DataVectorizer('./data/loc_map.pkl', './data/location_activity_map.pkl')
    model = VIMN_Lite(num_pois=len(vec.poi_vocab), num_acts=len(vec.act_vocab),
                      embed_dim=args.embed_dim, hidden_dim=args.hidden_dim)
    head = IntentContrastiveHead(args.hidden_dim)

    dataset = TrajectoryPairDataset(f'./data/{args.year}/{args.id}.pkl', num_neg=4)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=False)

    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    def encode_one(poi, act, time):
        with torch.set_grad_enabled(True):
            z = model(poi, act, time)
            z = head(z)
            z = nn.functional.normalize(z, dim=-1)
        return z

    for epoch in range(args.epochs):
        for anchor, positive, negs in loader:
            # DataLoader returns lists of strings; keep as Python lists
            batch_losses = []
            opt.zero_grad()
            all_loss = 0.0
            for i in range(len(anchor)):
                (poi_a, act_a, time_a), (poi_p, act_p, time_p), (poi_n, act_n, time_n) = vectorize_pair(
                    vec, anchor[i], positive[i], list(negs[i]))
                z_a = encode_one(poi_a, act_a, time_a)  # (1, H)
                z_p = encode_one(poi_p, act_p, time_p)  # (1, H)
                z_n = encode_one(poi_n, act_n, time_n)  # (N, H)

                # logits: [pos, neg1..negN]
                pos_sim = (z_a @ z_p.T)  # (1,1)
                neg_sim = (z_a @ z_n.T)  # (1,N)
                logits = torch.cat([pos_sim, neg_sim], dim=1)  # (1, 1+N)
                labels = torch.zeros((1,), dtype=torch.long)  # pos index 0
                loss = loss_fn(logits, labels)
                all_loss = all_loss + loss
            all_loss.backward()
            opt.step()
        print(f'Epoch {epoch+1}/{args.epochs} done')

    os.makedirs('./engine/experimental/checkpoints', exist_ok=True)
    torch.save({'model': model.state_dict(), 'head': head.state_dict()},
               './engine/experimental/checkpoints/vimn_lite.pt')
    print('Saved to ./engine/experimental/checkpoints/vimn_lite.pt')


if __name__ == '__main__':
    main()