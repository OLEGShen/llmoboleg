import os
import sys
import json
import pickle
import argparse
import subprocess


def run_cmd(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = p.communicate()
    code = p.returncode
    if code != 0:
        raise RuntimeError(f'Command failed: {cmd}\n{out.decode(errors="ignore")}')
    return out.decode(errors='ignore')


def read_pkl(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as f:
        return pickle.load(f)


def eval_pair(gen_map, gt_map, loc_cat):
    dates = sorted(set(gen_map.keys()) & set(gt_map.keys()))
    if not dates:
        return {'count': 0, 'cat_match': 0.0, 'time_match': 0.0, 'json_ok': 0.0}

    def route_to_cats(route):
        try:
            parts = route.split(': ')[-1]
            toks = [t.strip() for t in parts.split(',')]
            cats = []
            for t in toks:
                if '#' in t:
                    pid = int(t.split('#')[-1].split(' ')[0])
                    cat = None
                    if isinstance(loc_cat, dict):
                        cat = loc_cat.get(pid)
                    cats.append(cat)
                else:
                    cats.append(None)
            return cats
        except Exception:
            return []

    def parse_hour(route):
        try:
            return int(route.split(' at ')[-1][:2])
        except Exception:
            return None

    cat_hits = 0
    time_hits = 0
    json_ok = 0
    for d in dates:
        g = gen_map.get(d)
        r = gt_map.get(d)
        if g is None or r is None:
            continue
        gc = route_to_cats(g)
        rc = route_to_cats(r)
        m = min(len(gc), len(rc))
        cat_hits += sum(1 for i in range(m) if gc[i] is not None and rc[i] is not None and gc[i] == rc[i])
        gh = parse_hour(g)
        rh = parse_hour(r)
        if gh is not None and rh is not None and abs(gh - rh) <= 2:
            time_hits += 1
        json_ok += 1

    return {
        'count': len(dates),
        'cat_match': cat_hits / max(1, len(dates)),
        'time_match': time_hits / max(1, len(dates)),
        'json_ok': json_ok / max(1, len(dates)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', type=int, required=True)
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--dataset', type=str, default='2019')
    args = ap.parse_args()

    scenario_tag = {'2019': 'normal', '2021': 'abnormal', '20192021': 'normal_abnormal'}[args.dataset]

    cmd_base = f"python generate.py --dataset {args.dataset} --mode 0 --id {args.id} {'--fast' if args.fast else ''}"
    run_cmd(cmd_base)
    gen_dir = f"./result/{scenario_tag}/generated/llm_l/{args.id}/"
    gt_dir = f"./result/{scenario_tag}/ground_truth/llm_l/{args.id}/"
    gen_map_base = read_pkl(os.path.join(gen_dir, 'results.pkl'))
    gt_map = read_pkl(os.path.join(gt_dir, 'results.pkl'))
    with open(f"./data/{args.dataset}/{args.id}.pkl", 'rb') as f:
        att = pickle.load(f)
    loc_cat = att[11] if len(att) > 11 else {}
    m_base = eval_pair(gen_map_base, gt_map, loc_cat)

    cmd_vimn = f"python generate.py --dataset {args.dataset} --mode 0 --id {args.id} {'--fast' if args.fast else ''} --use_vimn"
    run_cmd(cmd_vimn)
    gen_map_v = read_pkl(os.path.join(gen_dir, 'results.pkl'))
    m_v = eval_pair(gen_map_v, gt_map, loc_cat)

    report = {
        'id': args.id,
        'dataset': args.dataset,
        'baseline': m_base,
        'vimn': m_v,
        'delta': {
            'cat_match': m_v['cat_match'] - m_base['cat_match'],
            'time_match': m_v['time_match'] - m_base['time_match'],
            'json_ok': m_v['json_ok'] - m_base['json_ok'],
        }
    }

    os.makedirs('./result/ab_test', exist_ok=True)
    out = f"./result/ab_test/single_{args.dataset}_{args.id}.json"
    with open(out, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('Saved', out)


if __name__ == '__main__':
    main()