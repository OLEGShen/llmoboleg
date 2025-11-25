import os
import sys
import pickle
import torch

sys.path.append(os.path.abspath('.'))
from engine.experimental.wrapper import build_vimn
from engine.experimental.memento import MementoCase, MementoMemory, MementoPolicy, MementoAgent


def main():
    year = '2019'
    pid = '934'
    pkl = f'./data/{year}/{pid}.pkl'
    with open(pkl, 'rb') as f:
        att = pickle.load(f)
    train_trajs = att[0]
    test_trajs = att[1]

    vec, model = build_vimn('./data/loc_map.pkl', './data/location_activity_map.pkl', 64, 128)
    head = None
    ckpt = './engine/experimental/checkpoints/vimn_lite.pt'
    from engine.experimental.vimn import IntentContrastiveHead
    head = IntentContrastiveHead(128)
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state['model'])
        head.load_state_dict(state['head'])

    memory = MementoMemory()
    for t in train_trajs[:20]:
        date_str = t.split(': ')[0].split(' ')[-1]
        case = MementoCase(t, date_str, vec, model, head)
        memory.add(case)

    agent = MementoAgent(vec, model, head, memory, MementoPolicy())
    query = test_trajs[0]
    idxs = agent.step(query, top_k=3)
    print('Selected indices:', idxs)
    for i in idxs:
        print('Case:', memory.cases[i].date, memory.cases[i].traj)
    agent.update(query, idxs)
    print('Updated rewards:', [memory.cases[i].reward for i in idxs])


if __name__ == '__main__':
    main()