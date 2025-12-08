import os
import argparse
import pickle
import subprocess
import math
import torch
import sys
from datetime import datetime

sys.path.append(os.path.abspath('.'))
from engine.dpo_gating import DPOTrainer


def run_cmd(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = p.communicate()
    code = p.returncode
    if code != 0:
        raise RuntimeError(out.decode(errors='ignore'))
    return out.decode(errors='ignore')


def read_pkl(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as f:
        return pickle.load(f)


def first_loc(route):
    try:
        part = route.split(': ', 1)[1]
        seg = part.split(',')[0].strip()
        import re
        m = re.search(r"(.+?)#(\d+)", seg)
        if m:
            return m.group(1).strip(), int(m.group(2))
        toks = part.replace(', ', ' at ').split(' at ')
        loc = toks[0].strip()
        name = loc.split('#')[0].strip()
        pid = int(loc.split('#')[-1]) if '#' in loc else None
        return name, pid
    except Exception:
        return None, None


def check_poi_match(gen_text, gt_text):
    gn, gi = first_loc(gen_text)
    rn, ri = first_loc(gt_text)
    if gi is not None and ri is not None:
        return int(gi) == int(ri)
    if gn is None or rn is None:
        return False
    return str(gn).strip() == str(rn).strip()


def compute_entropy_from_topk(topk):
    if not isinstance(topk, list) or len(topk) == 0:
        return 1.0
    ps = [float(d.get('prob', 0.0)) for d in topk]
    s = sum(ps)
    if s <= 0:
        return 1.0
    p = [x / s for x in ps]
    H = sum([-pi * math.log(max(1e-8, pi)) for pi in p])
    Hmax = math.log(len(p))
    return float(H / max(1e-8, Hmax))


def time_embedding_from_date(date_str):
    try:
        y, m, d = map(int, date_str.split('-'))
        wd = datetime(y, m, d).weekday()
    except Exception:
        wd = 0
    onehot = [1.0 if i == wd else 0.0 for i in range(7)]
    hour = 0
    emb = onehot + [math.sin(2 * math.pi * hour / 24.0), math.cos(2 * math.pi * hour / 24.0)]
    if len(emb) < 32:
        emb += [0.0] * (32 - len(emb))
    return emb[:32]


def extract_memento_score(d_detail, strict=True):
    if not isinstance(d_detail, dict):
        raise RuntimeError('details_m missing')
    for k in ['memento_top1_score', 'top1_score', 'memento_strength']:
        v = d_detail.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    if strict:
        raise RuntimeError('memento_top1_score not found in details_m')
    return 0.5


def build_preference_pairs(dataset, pid, fast=True, strict_memento_score=True):
    scenario_tag = {'2019': 'normal', '2021': 'abnormal', '20192021': 'normal_abnormal'}[dataset]
    gen_dir = f"./result/{scenario_tag}/generated/llm_l/{pid}/"
    gt_dir = f"./result/{scenario_tag}/ground_truth/llm_l/{pid}/"

    cmd_v = f"python generate.py --dataset {dataset} --mode 0 --id {pid} {'--fast' if fast else ''} --use_vimn"
    run_cmd(cmd_v)
    details_v = read_pkl(os.path.join(gen_dir, 'details.pkl'))
    results_v = read_pkl(os.path.join(gen_dir, 'results.pkl'))

    cmd_m = f"python generate.py --dataset {dataset} --mode 0 --id {pid} {'--fast' if fast else ''} --use_memento"
    run_cmd(cmd_m)
    details_m = read_pkl(os.path.join(gen_dir, 'details.pkl'))
    results_m = read_pkl(os.path.join(gen_dir, 'results.pkl'))

    cmd_h = f"python generate.py --dataset {dataset} --mode 0 --id {pid} {'--fast' if fast else ''} --use_vimn --use_memento"
    run_cmd(cmd_h)
    details_h = read_pkl(os.path.join(gen_dir, 'details.pkl'))
    results_h = read_pkl(os.path.join(gen_dir, 'results.pkl'))

    gt_map = read_pkl(os.path.join(gt_dir, 'results.pkl'))

    dates = sorted(set(results_v.keys()) & set(results_m.keys()) & set(results_h.keys()) & set(gt_map.keys()))
    pairs = []
    for d in dates:
        gv = results_v.get(d)
        gm = results_m.get(d)
        gh = results_h.get(d)
        gr = gt_map.get(d)
        if gv is None or gm is None or gh is None or gr is None:
            continue

        dv = details_v.get(d, {}) if isinstance(details_v, dict) else {}
        dm = details_m.get(d, {}) if isinstance(details_m, dict) else {}
        ent = compute_entropy_from_topk(dv.get('vimn_topk', []))
        emb = time_embedding_from_date(d)
        ms = extract_memento_score(dm, strict=strict_memento_score)
        state = torch.tensor([ent, ms] + emb, dtype=torch.float32)

        cv = check_poi_match(gv, gr)
        cm = check_poi_match(gm, gr)
        ch = check_poi_match(gh, gr)

        winner = None
        if cv:
            winner = 0
        elif cm:
            winner = 1
        elif (not cv and not cm) and ch:
            winner = 2
        else:
            winner = None

        if winner is None:
            continue

        losers = []
        if not cv:
            losers.append(0)
        if not cm:
            losers.append(1)
        if not ch:
            losers.append(2)
        losers = [l for l in losers if l != winner]
        for l in losers:
            pairs.append((state, winner, l))
    return pairs


def load_user_ids_from_file(path):
    ids = []
    if not path or not os.path.exists(path):
        return ids
    with open(path, 'r') as f:
        text = f.read()
    for tok in text.replace('\n', ',').split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            ids.append(int(tok))
        except Exception:
            pass
    return sorted(set(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='2019')
    ap.add_argument('--id', type=int, default=None)
    ap.add_argument('--ids', type=str, default=None)
    ap.add_argument('--user_list_file', type=str, default=None)
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--beta', type=float, default=0.1)
    ap.add_argument('--cost_beta', type=float, default=0.1)
    ap.add_argument('--save_path', type=str, default=None)
    ap.add_argument('--allow_score_fallback', action='store_true')
    args = ap.parse_args()

    id_list = []
    if args.user_list_file:
        id_list = load_user_ids_from_file(args.user_list_file)
    elif args.ids:
        for tok in args.ids.split(','):
            tok = tok.strip()
            if not tok:
                continue
            try:
                id_list.append(int(tok))
            except Exception:
                pass
        id_list = sorted(set(id_list))
    elif args.id is not None:
        id_list = [args.id]

    all_pairs = []
    for pid in id_list:
        user_pairs = build_preference_pairs(args.dataset, pid, fast=args.fast, strict_memento_score=(not args.allow_score_fallback))
        all_pairs.extend(user_pairs)

    trainer = DPOTrainer(lr=args.lr, beta=args.beta, cost_beta=args.cost_beta)
    for _ in range(max(1, args.epochs)):
        for state, win_idx, lose_idx in all_pairs:
            trainer.compute_loss(state, win_idx, lose_idx)

    save_dir = './engine/experimental/checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    name = 'gating_dpo_multi' if len(id_list) > 1 else f'gating_dpo_{id_list[0]}' if id_list else 'gating_dpo'
    out = args.save_path or os.path.join(save_dir, f'{name}.pt')
    torch.save(trainer.model.state_dict(), out)
    print(f'Saved gating checkpoint to {out}')


if __name__ == '__main__':
    main()