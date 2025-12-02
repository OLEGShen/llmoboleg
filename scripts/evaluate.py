import os
import sys
import json
import pickle
import argparse
import re
import math
import numpy as np
sys.path.append(os.path.abspath('.'))
from engine.vimn_loader import load_vimn_gru_ckpt

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

def build_geo_map(vec):
    m = {}
    for k, v in vec.loc_map.items():
        try:
            pid = int(str(v).split('#')[-1])
            mm = re.search(r"\(([-0-9\.]+),\s*([-0-9\.]+)\)", str(k))
            if mm:
                lat = float(mm.group(1)); lon = float(mm.group(2))
                m[pid] = (lat, lon)
        except Exception:
            continue
    return m

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c

def parse_plan_items(route_str):
    items = []
    if not isinstance(route_str, str):
        return items
    parts = route_str.split(': ', 1)
    if len(parts) < 2:
        return items
    tokens = [t.strip() for t in parts[1].split(',')]
    for tok in tokens:
        m = re.search(r"#(\d+)\s+at\s+(\d{2}):(\d{2})(?::\d{2})?", tok)
        if m:
            pid = int(m.group(1))
            hh = int(m.group(2)); mm = int(m.group(3))
            items.append((pid, hh*60+mm))
    return items

def step_distance_series(route_str, id2geo):
    seq = parse_plan_items(route_str)
    out = []
    for i in range(1, len(seq)):
        p1 = id2geo.get(seq[i-1][0]); p2 = id2geo.get(seq[i][0])
        if p1 is None or p2 is None:
            continue
        d = haversine(p1[0], p1[1], p2[0], p2[1])
        out.append(d)
    return out

def step_interval_series(route_str):
    seq = parse_plan_items(route_str)
    out = []
    for i in range(1, len(seq)):
        dt = seq[i][1] - seq[i-1][1]
        if dt >= 0:
            out.append(dt)
    return out

def hist_prob(values, bins):
    if len(values) == 0:
        return np.zeros(len(bins)-1)
    h, _ = np.histogram(values, bins=bins)
    h = h.astype(np.float64)
    s = h.sum()
    if s == 0:
        return h
    return h/s

def jsd(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    p /= p.sum(); q /= q.sum()
    m = 0.5*(p+q)
    def kl(a, b):
        return np.sum(a*np.log(a/b))
    return 0.5*kl(p, m) + 0.5*kl(q, m)

def dard_distribution(route_str, vec, time_bins=144):
    seq = parse_plan_items(route_str)
    counts = np.zeros((time_bins, len(vec.act_vocab)), dtype=np.float64)
    for pid, tmin in seq:
        tb = (tmin//60) % time_bins
        cat = vec.id2act.get(pid, 'UNK')
        ai = vec.act_vocab.get(cat, vec.act_vocab.get('UNK'))
        counts[tb, ai] += 1.0
    s = counts.sum()
    if s > 0:
        counts /= s
    return counts.reshape(-1)

def stvd_distribution(route_str, id2geo, time_bins=144, lat_bins=20, lon_bins=20):
    seq = parse_plan_items(route_str)
    if len(seq) == 0 or len(id2geo) == 0:
        return np.zeros(time_bins*lat_bins*lon_bins)
    lats = [id2geo[pid][0] for pid in id2geo]
    lons = [id2geo[pid][1] for pid in id2geo]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    lat_edges = np.linspace(lat_min, lat_max, lat_bins+1)
    lon_edges = np.linspace(lon_min, lon_max, lon_bins+1)
    counts = np.zeros((time_bins, lat_bins, lon_bins), dtype=np.float64)
    for pid, tmin in seq:
        geo = id2geo.get(pid)
        if geo is None:
            continue
        tb = (tmin//60) % time_bins
        li = np.clip(np.digitize([geo[0]], lat_edges)[0]-1, 0, lat_bins-1)
        lj = np.clip(np.digitize([geo[1]], lon_edges)[0]-1, 0, lon_bins-1)
        counts[tb, li, lj] += 1.0
    s = counts.sum()
    if s > 0:
        counts /= s
    return counts.reshape(-1)

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

    vec = None
    try:
        ckpt_global = './engine/experimental/checkpoints/vimn_global_gru_2019.pt' if args.dataset == '2019' else ('./engine/experimental/checkpoints/vimn_global_gru_2021.pt' if args.dataset == '2021' else './engine/experimental/checkpoints/vimn_global_gru_20192021.pt')
        if os.path.exists(ckpt_global):
            vec, _, _ = load_vimn_gru_ckpt(ckpt_global)
    except Exception:
        vec = None
    id2geo = build_geo_map(vec) if vec is not None else {}

    sd_base = []
    sd_vimn = []
    sd_gt = []
    si_base = []
    si_vimn = []
    si_gt = []
    dard_base = None
    dard_vimn = None
    dard_gt = None
    stvd_base = None
    stvd_vimn = None
    stvd_gt = None
    dates = sorted(set(gen_map_base.keys()) & set(gen_map_vimn.keys()) & set(gt_map.keys()))
    for d in dates:
        rb = gen_map_base.get(d); rv = gen_map_vimn.get(d); rg = gt_map.get(d)
        sd_base += step_distance_series(rb, id2geo)
        sd_vimn += step_distance_series(rv, id2geo)
        sd_gt += step_distance_series(rg, id2geo)
        si_base += step_interval_series(rb)
        si_vimn += step_interval_series(rv)
        si_gt += step_interval_series(rg)
        if vec is not None:
            db = dard_distribution(rb, vec)
            dv = dard_distribution(rv, vec)
            dg = dard_distribution(rg, vec)
            dard_base = db if dard_base is None else dard_base + db
            dard_vimn = dv if dard_vimn is None else dard_vimn + dv
            dard_gt = dg if dard_gt is None else dard_gt + dg
        sb = stvd_distribution(rb, id2geo)
        sv = stvd_distribution(rv, id2geo)
        sg = stvd_distribution(rg, id2geo)
        stvd_base = sb if stvd_base is None else stvd_base + sb
        stvd_vimn = sv if stvd_vimn is None else stvd_vimn + sv
        stvd_gt = sg if stvd_gt is None else stvd_gt + sg

    sd_bins = np.linspace(0.0, max(sd_gt + sd_base + sd_vimn + [1.0]), 21)
    si_bins = np.linspace(0.0, max(si_gt + si_base + si_vimn + [60.0]), 25)
    sd_p_base = hist_prob(sd_base, sd_bins)
    sd_p_vimn = hist_prob(sd_vimn, sd_bins)
    sd_p_gt = hist_prob(sd_gt, sd_bins)
    si_p_base = hist_prob(si_base, si_bins)
    si_p_vimn = hist_prob(si_vimn, si_bins)
    si_p_gt = hist_prob(si_gt, si_bins)
    jsd_sd_base = jsd(sd_p_base, sd_p_gt)
    jsd_sd_vimn = jsd(sd_p_vimn, sd_p_gt)
    jsd_si_base = jsd(si_p_base, si_p_gt)
    jsd_si_vimn = jsd(si_p_vimn, si_p_gt)
    jsd_dard_base = jsd(dard_base, dard_gt) if (dard_base is not None and dard_gt is not None) else None
    jsd_dard_vimn = jsd(dard_vimn, dard_gt) if (dard_vimn is not None and dard_gt is not None) else None
    jsd_stvd_base = jsd(stvd_base, stvd_gt) if (stvd_base is not None and stvd_gt is not None) else None
    jsd_stvd_vimn = jsd(stvd_vimn, stvd_gt) if (stvd_vimn is not None and stvd_gt is not None) else None

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
    print("-"*40)
    print("Step Distance (km):")
    print(f"  Baseline mean: {np.mean(sd_base) if len(sd_base)>0 else 0.0:.4f}  median: {np.median(sd_base) if len(sd_base)>0 else 0.0:.4f}  JSD vs GT: {jsd_sd_base:.4f}")
    print(f"  VIMN     mean: {np.mean(sd_vimn) if len(sd_vimn)>0 else 0.0:.4f}  median: {np.median(sd_vimn) if len(sd_vimn)>0 else 0.0:.4f}  JSD vs GT: {jsd_sd_vimn:.4f}")
    print("Step Interval (min):")
    print(f"  Baseline mean: {np.mean(si_base) if len(si_base)>0 else 0.0:.4f}  median: {np.median(si_base) if len(si_base)>0 else 0.0:.4f}  JSD vs GT: {jsd_si_base:.4f}")
    print(f"  VIMN     mean: {np.mean(si_vimn) if len(si_vimn)>0 else 0.0:.4f}  median: {np.median(si_vimn) if len(si_vimn)>0 else 0.0:.4f}  JSD vs GT: {jsd_si_vimn:.4f}")
    if jsd_dard_base is not None:
        print("DARD Jensen-Shannon divergence:")
        print(f"  Baseline: {jsd_dard_base:.4f}  VIMN: {jsd_dard_vimn:.4f}")
    if jsd_stvd_base is not None:
        print("STVD Jensen-Shannon divergence:")
        print(f"  Baseline: {jsd_stvd_base:.4f}  VIMN: {jsd_stvd_vimn:.4f}")
    delta = acc_vimn - acc_base
    improvement = (delta / acc_base * 100) if acc_base > 0 else float('inf')
    print(f"Delta (VIMN - Baseline): {delta:+.4f}")
    if delta > 0:
        print(f"Relative Improvement: {improvement:+.2f}%")
    print("="*40 + "\n")

if __name__ == '__main__':
    main()