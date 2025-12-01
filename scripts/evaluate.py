import os
import sys
import json
import pickle
import argparse
import re

def read_pkl(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def parse_loc_ids(route_str):
    """Parses a trajectory string to extract a list of location IDs."""
    if not isinstance(route_str, str):
        return []
    try:
        # Find the part after the date, e.g., "Home #1 at 08:00, Office #2 at 09:00, ..."
        plan_part = route_str.split(': ', 1)[-1]
        # Use regex to find all occurrences of '#<number>'
        loc_ids = re.findall(r'#(\d+)', plan_part)
        return [int(pid) for pid in loc_ids]
    except Exception as e:
        # print(f"Could not parse route string: {route_str}, Error: {e}")
        return []

def parse_first_poi_id(route_str):
    if not isinstance(route_str, str):
        return None
    try:
        m = re.search(r'#(\d+)', route_str)
        if m:
            return int(m.group(1))
        return None
    except Exception:
        return None

def compute_acck(gen_map, gt_map, ks=(1, 5, 10)):
    counts = {k: {'hits': 0, 'total': 0} for k in ks}
    dates = sorted(set(gen_map.keys()) & set(gt_map.keys()))
    for d in dates:
        gen_route = gen_map.get(d)
        gt_route = gt_map.get(d)
        pred_list = parse_loc_ids(gen_route)
        gt_pid = parse_first_poi_id(gt_route)
        if gt_pid is None or len(pred_list) == 0:
            continue
        for k in ks:
            topk = pred_list[:k]
            if gt_pid in topk:
                counts[k]['hits'] += 1
            counts[k]['total'] += 1
    return {f'Acc@{k}': (counts[k]['hits'] / counts[k]['total'] if counts[k]['total'] > 0 else 0.0) for k in ks}

def calculate_loc_acc(gen_map, gt_map):
    """Calculates Location Accuracy (Loc-ACC)."""
    total_hits = 0
    total_gt_locs = 0

    dates = sorted(set(gen_map.keys()) & set(gt_map.keys()))
    if not dates:
        return 0.0, 0, 0

    for d in dates:
        gen_route = gen_map.get(d)
        gt_route = gt_map.get(d)

        gen_locs = parse_loc_ids(gen_route)
        gt_locs = parse_loc_ids(gt_route)

        if not gt_locs:
            continue

        total_gt_locs += len(gt_locs)
        
        # Compare element-wise up to the minimum length of the two sequences
        min_len = min(len(gen_locs), len(gt_locs))
        for i in range(min_len):
            if gen_locs[i] == gt_locs[i]:
                total_hits += 1
    
    accuracy = total_hits / total_gt_locs if total_gt_locs > 0 else 0.0
    return accuracy, total_hits, total_gt_locs

def main():
    parser = argparse.ArgumentParser(description="Evaluate trajectory generation results with Loc-ACC.")
    parser.add_argument('--id', type=int, required=True, help='Agent ID to evaluate.')
    parser.add_argument('--dataset', type=str, default='2019', help='Dataset used for the test (e.g., 2019).')
    args = parser.parse_args()

    ab_test_dir = './result/ab_test'
    exp_dir = os.path.join(ab_test_dir, f'exp_{args.dataset}_{args.id}') if args.dataset != '2019' else os.path.join(ab_test_dir, f'exp_2019_{args.id}')
    scenario_tag = {'2019': 'normal', '2021': 'abnormal', '20192021': 'normal_abnormal'}[args.dataset]
    gt_dir = f"./result/{scenario_tag}/ground_truth/llm_l/{args.id}/"

    # --- Load Data ---
    baseline_path = os.path.join(ab_test_dir, f'single_{args.dataset}_{args.id}_baseline_results.pkl')
    vimn_path = os.path.join(ab_test_dir, f'single_{args.dataset}_{args.id}_vimn_results.pkl')
    # Prefer new exp_dir copies if available
    if os.path.exists(os.path.join(exp_dir, 'baseline_details.pkl')):
        baseline_path = os.path.join(exp_dir, 'baseline_results.pkl') if os.path.exists(os.path.join(exp_dir, 'baseline_results.pkl')) else baseline_path
        vimn_path = os.path.join(exp_dir, 'vimn_results.pkl') if os.path.exists(os.path.join(exp_dir, 'vimn_results.pkl')) else vimn_path
    gt_path = os.path.join(gt_dir, 'results.pkl')

    gen_map_base = read_pkl(baseline_path)
    gen_map_vimn = read_pkl(vimn_path)
    gt_map = read_pkl(gt_path)

    if gen_map_base is None or gen_map_vimn is None or gt_map is None:
        print("\nOne or more result files are missing. Please run the A/B test first:")
        print(f"python scripts/run_single_ab_test.py --dataset {args.dataset} --id {args.id}")
        return

    # --- Calculate Metrics ---
    acc_base, hits_base, total_locs = calculate_loc_acc(gen_map_base, gt_map)
    acc_vimn, hits_vimn, _ = calculate_loc_acc(gen_map_vimn, gt_map)
    # Acc@k using LLM generated candidate locations (independent of VIMN output)
    acck_base = compute_acck(gen_map_base, gt_map, ks=(1, 5, 10))
    acck_vimn = compute_acck(gen_map_vimn, gt_map, ks=(1, 5, 10))

    # --- Print Report ---
    print("\n" + "="*40)
    print(f"Location Accuracy (Loc-ACC) Evaluation")
    print(f"Agent ID: {args.id}, Dataset: {args.dataset}")
    print("-"*40)
    print(f"Baseline Loc-ACC: {acc_base:.4f} ({hits_base}/{total_locs} correct locations)")
    print(f"VIMN Loc-ACC:     {acc_vimn:.4f} ({hits_vimn}/{total_locs} correct locations)")
    print("-"*40)
    print("Baseline Acc@1/5/10:")
    print(f"  Acc@1: {acck_base.get('Acc@1', 0.0):.4f}  Acc@5: {acck_base.get('Acc@5', 0.0):.4f}  Acc@10: {acck_base.get('Acc@10', 0.0):.4f}")
    print("VIMN Acc@1/5/10:")
    print(f"  Acc@1: {acck_vimn.get('Acc@1', 0.0):.4f}  Acc@5: {acck_vimn.get('Acc@5', 0.0):.4f}  Acc@10: {acck_vimn.get('Acc@10', 0.0):.4f}")
    delta = acc_vimn - acc_base
    improvement = (delta / acc_base * 100) if acc_base > 0 else float('inf')
    print(f"Delta (VIMN - Baseline): {delta:+.4f}")
    if delta > 0:
        print(f"Relative Improvement: {improvement:+.2f}%")
    print("="*40 + "\n")

if __name__ == '__main__':
    main()