import os
import sys
import pickle
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=str, default='2019')
    parser.add_argument('--ids', type=str, required=True, help='逗号分隔的id列表，如 101,102,934')
    parser.add_argument('--out', type=str, default='./data/2019/all.pkl')
    parser.add_argument('--train_per_user', type=int, default=0)
    parser.add_argument('--test_per_user', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_freq', type=int, default=2)
    args = parser.parse_args()

    ids = [s.strip() for s in args.ids.split(',') if s.strip()]
    train_trajs = []
    test_trajs = []
    used_poi_ids = set()
    poi_freq = {}
    used_act_names = set()
    import random
    random.seed(args.seed)
    for pid in ids:
        p = f'./data/{args.year}/{pid}.pkl'
        if not os.path.exists(p):
            print('skip missing', p)
            continue
        with open(p, 'rb') as f:
            att = pickle.load(f)
        if isinstance(att, (list, tuple)) and len(att) >= 2:
            if isinstance(att[0], (list, tuple)):
                bucket = list(att[0])
                if args.train_per_user and args.train_per_user > 0:
                    random.shuffle(bucket)
                    bucket = bucket[: min(args.train_per_user, len(bucket))]
                train_trajs.extend(bucket)
            if isinstance(att[1], (list, tuple)):
                bucket = list(att[1])
                if args.test_per_user and args.test_per_user > 0:
                    random.shuffle(bucket)
                    bucket = bucket[: min(args.test_per_user, len(bucket))]
                test_trajs.extend(bucket)
            for bucket in [att[0], att[1]]:
                if isinstance(bucket, (list, tuple)):
                    for route in bucket:
                        try:
                            parts = route.split(': ')[-1]
                            tokens = [t.strip() for t in parts.split(',')]
                            for tok in tokens:
                                m = __import__('re').match(r"^(.*?)#(\d+)\s+at\s+(\d{2}):(\d{2})", tok)
                                if m:
                                    pid = int(m.group(2))
                                    used_poi_ids.add(pid)
                                    poi_freq[pid] = poi_freq.get(pid, 0) + 1
                        except Exception:
                            pass

    filtered_poi = [pid for pid in sorted(used_poi_ids) if poi_freq.get(pid, 0) >= args.min_freq]
    out = (train_trajs, test_trajs, filtered_poi, None)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(out, f)
    print('saved', args.out, 'train', len(train_trajs), 'test', len(test_trajs), 'poi_ids', len(filtered_poi))

if __name__ == '__main__':
    main()