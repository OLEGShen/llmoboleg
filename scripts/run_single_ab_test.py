import os
import sys
import json
import pickle
import argparse
import subprocess
import shutil
import torch

sys.path.append(os.path.abspath('.'))
from engine.experimental.vimn import DataVectorizer
from engine.vimn_core import VIMN
from engine.vimn_loader import load_vimn_gru_ckpt


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


def eval_pair(gen_map, gt_map, vec):
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
                    cat = vec.id2act.get(pid, 'UNK') if vec is not None else None
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
    cat_total = 0
    for d in dates:
        g = gen_map.get(d)
        r = gt_map.get(d)
        if g is None or r is None:
            continue
        gc = route_to_cats(g)
        rc = route_to_cats(r)
        m = min(len(gc), len(rc))
        if m == 0:
            continue
        cat_total += m
        cat_hits += sum(1 for i in range(m) if gc[i] is not None and rc[i] is not None and gc[i] == rc[i])
        gh = parse_hour(g)
        rh = parse_hour(r)
        if gh is not None and rh is not None and abs(gh - rh) <= 2:
            time_hits += 1
        json_ok += 1

    return {
        'count': len(dates),
        'cat_match': cat_hits / max(1, cat_total),
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

    vec = None
    try:
        if scenario_tag == 'normal':
            ckpt_global = './engine/experimental/checkpoints/vimn_global_gru_2019.pt'
        elif scenario_tag == 'abnormal':
            ckpt_global = './engine/experimental/checkpoints/vimn_global_gru_2021.pt'
        else:
            ckpt_global = './engine/experimental/checkpoints/vimn_global_gru_20192021.pt'
        if os.path.exists(ckpt_global):
            vec, _, _ = load_vimn_gru_ckpt(ckpt_global)
            print(f"Loaded vectorizer from {ckpt_global}")
    except Exception as e:
        print(f"Warning: Could not load VIMN ckpt to get vectorizer. Cat match may be inaccurate. Error: {e}")

    gen_dir = f"./result/{scenario_tag}/generated/llm_l/{args.id}/"
    gt_dir = f"./result/{scenario_tag}/ground_truth/llm_l/{args.id}/"
    report_dir = './result/ab_test'
    os.makedirs(report_dir, exist_ok=True)
    exp_dir = os.path.join(report_dir, f"exp_2019_{args.id}")
    os.makedirs(exp_dir, exist_ok=True)

    # --- Baseline
    cmd_base = f"python generate.py --dataset {args.dataset} --mode 0 --id {args.id} {'--fast' if args.fast else ''}"
    print(f"\nRunning baseline: {cmd_base}")
    run_cmd(cmd_base)
    gen_result_path = os.path.join(gen_dir, 'results.pkl')
    baseline_output_path = os.path.join(report_dir, f"single_{args.dataset}_{args.id}_baseline_results.pkl")
    details_path = os.path.join(gen_dir, 'details.pkl')
    if os.path.exists(gen_result_path):
        shutil.copy(gen_result_path, baseline_output_path)
        print(f"Saved baseline output to {baseline_output_path}")
        try:
            if os.path.exists(details_path):
                shutil.copy(details_path, os.path.join(exp_dir, 'baseline_details.pkl'))
                print(f"Saved baseline details to {os.path.join(exp_dir, 'baseline_details.pkl')}")
            shutil.copy(gen_result_path, os.path.join(exp_dir, 'baseline_results.pkl'))
        except Exception:
            pass
    gen_map_base = read_pkl(gen_result_path)
    gt_map = read_pkl(os.path.join(gt_dir, 'results.pkl'))

    # --- VIMN
    cmd_vimn = f"python generate.py --dataset {args.dataset} --mode 0 --id {args.id} {'--fast' if args.fast else ''} --use_vimn"
    print(f"\nRunning VIMN: {cmd_vimn}")
    run_cmd(cmd_vimn)
    vimn_output_path = os.path.join(report_dir, f"single_{args.dataset}_{args.id}_vimn_results.pkl")
    if os.path.exists(gen_result_path):
        shutil.copy(gen_result_path, vimn_output_path)
        print(f"Saved VIMN output to {vimn_output_path}")
        try:
            if os.path.exists(details_path):
                shutil.copy(details_path, os.path.join(exp_dir, 'vimn_details.pkl'))
                print(f"Saved VIMN details to {os.path.join(exp_dir, 'vimn_details.pkl')}")
            shutil.copy(gen_result_path, os.path.join(exp_dir, 'vimn_results.pkl'))
        except Exception:
            pass
    gen_map_v = read_pkl(gen_result_path)

    # --- Evaluation
    print("\nEvaluating results...")
    m_base = eval_pair(gen_map_base, gt_map, vec)
    m_v = eval_pair(gen_map_v, gt_map, vec)

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

    out_json = f"{exp_dir}/single_{args.dataset}_{args.id}.json"
    with open(out_json, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('Saved report to', out_json)


if __name__ == '__main__':
    main()