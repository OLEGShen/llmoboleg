import argparse
import pickle
import torch
import os
import sys

# 保证从项目根目录执行时可以找到 engine 包
sys.path.append(os.path.abspath('.'))
from engine.experimental.wrapper import build_vimn, encode_intent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=str, default='2019')
    parser.add_argument('--id', type=str, default='934')
    parser.add_argument('--only_test', action='store_true', help='只用一条测试轨迹生成向量')
    args = parser.parse_args()

    vec, model = build_vimn(
        loc_map_path="./data/loc_map.pkl",
        act_map_path="./data/location_activity_map.pkl",
        embed_dim=64,
        hidden_dim=128
    )

    pkl_path = f"./data/{args.year}/{args.id}.pkl"
    with open(pkl_path, 'rb') as f:
        att = pickle.load(f)

    trajs = []
    if args.only_test:
        if isinstance(att[1], (list, tuple)) and len(att[1]) > 0:
            trajs.append(att[1][0])
        else:
            raise ValueError('测试轨迹不存在')
    else:
        if isinstance(att[0], (list, tuple)) and len(att[0]) > 0:
            trajs.append(att[0][0])
        if isinstance(att[1], (list, tuple)) and len(att[1]) > 0:
            trajs.append(att[1][0])

    poi_ids, act_ids, time_ids = vec.vectorize_sequence(trajs)
    print('poi_ids', poi_ids.shape, poi_ids.dtype)
    print('act_ids', act_ids.shape, act_ids.dtype)
    print('time_ids', time_ids.shape, time_ids.dtype)

    with torch.no_grad():
        intent_vec = model(poi_ids, act_ids, time_ids)
    print('intent_vec', intent_vec.shape)
    assert intent_vec.ndim == 2 and intent_vec.shape[1] == 128, '意图向量维度不匹配'


if __name__ == '__main__':
    main()