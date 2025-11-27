import os
import sys
import pickle
import torch

sys.path.append(os.path.abspath('.'))
from engine.experimental.wrapper import build_vimn
from engine.experimental.intent_retriever import IntentRetriever


def main():
    year = '2019'
    pid = '934'
    pkl = f'./data/{year}/{pid}.pkl'
    with open(pkl, 'rb') as f:
        att = pickle.load(f)
    train_trajs = att[0]
    test_trajs = att[1]

    allowed_poi = att[2] if len(att) >= 3 else None
    allowed_act = att[3] if len(att) >= 4 else None
    vec, model = build_vimn('./data/loc_map.pkl', './data/location_activity_map.pkl', 64, 128,
                            allowed_poi_ids=allowed_poi, allowed_act_names=allowed_act)
    ckpt_path = './engine/experimental/checkpoints/vimn_lite.pt'
    head = None
    if os.path.exists(ckpt_path):
        from engine.experimental.vimn import IntentContrastiveHead
        head = IntentContrastiveHead(128)
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        head.load_state_dict(state['head'])
    else:
        from engine.experimental.vimn import IntentContrastiveHead
        head = IntentContrastiveHead(128)

    retr = IntentRetriever(vec, model, head, train_trajs, top_k=3, test_trajs=test_trajs)

    # 取第一条测试日期，按意图检索 Top-3 历史
    query_date = test_trajs[0].split(': ')[0].split(' ')[-1]
    top = retr.retrieve(query_date)
    print('Query date:', query_date)
    print('Retrieved (Top-3):')
    for t in top:
        print('-', t)


if __name__ == '__main__':
    main()