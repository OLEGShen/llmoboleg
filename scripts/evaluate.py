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
        plan_part = route_str.split(': ', 1)[-1]
        loc_ids = re.findall(r'#(\d+)', plan_part)
        return [int(pid) for pid in loc_ids]
    except Exception as e:
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
        min_len = min(len(gen_locs), len(gt_locs))
        for i in range(min_len):
            if gen_locs[i] == gt_locs[i]:
                total_hits += 1
    accuracy = total_hits / total_gt_locs if total_gt_locs > 0 else 0.0
    return accuracy, total_hits, total_gt_locs

def build_geo_map(vec):
    m = {}
    if not vec: return m
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

def dard_distribution(route_str, vec, time_bins=24):
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

def stvd_distribution(route_str, id2geo, time_bins=24, lat_bins=20, lon_bins=20):
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
    parser = argparse.ArgumentParser(description="Evaluate trajectory generation results.")
    parser.add_argument('--id', type=int, required=True, help='Agent ID to evaluate.')
    parser.add_argument('--dataset', type=str, default='2019', help='Dataset used for the test.')
    parser.add_argument('--exp_dir', type=str, required=True, help='Directory containing experiment result files (*_results.pkl).')
    args = parser.parse_args()

    scenario_tag = {'2019': 'normal', '2021': 'abnormal', '20192021': 'normal_abnormal'}.get(args.dataset, 'normal')
    gt_dir = f"./result/{scenario_tag}/ground_truth/llm_l/{args.id}/"
    gt_path = os.path.join(gt_dir, 'results.pkl')
    gt_map = read_pkl(gt_path)
    if gt_map is None:
        print(f"Ground truth file not found at {gt_path}")
        return

    # --- Load VEC and GEO data ---
    vec = None
    try:
        ckpt_global = f'./engine/experimental/checkpoints/vimn_global_gru_{args.dataset}.pt'
        if os.path.exists(ckpt_global):
            vec, _, _ = load_vimn_gru_ckpt(ckpt_global)
    except Exception as e:
        print(f"Could not load VIMN vectorizer: {e}")
    id2geo = build_geo_map(vec) if vec is not None else {}

    # --- Dynamically Load Experiment Data ---
    exp_results = {}
    try:
        for f in os.listdir(args.exp_dir):
            if f.endswith("_results.pkl"):
                name = f.replace("_results.pkl", "")
                path = os.path.join(args.exp_dir, f)
                data = read_pkl(path)
                if data:
                    exp_results[name] = data
    except FileNotFoundError:
        print(f"Error: Experiment directory not found at {args.exp_dir}")
        return
    
    if not exp_results:
        print(f"No '*_results.pkl' files found in {args.exp_dir}")
        return

    # --- Calculate Metrics for each experiment ---
    all_metrics = {}
    common_dates = set(gt_map.keys())
    for name, gen_map in exp_results.items():
        common_dates &= set(gen_map.keys())
    dates = sorted(list(common_dates))

    for name, gen_map in exp_results.items():
        metrics = {}
        acc, hits, total_locs = calculate_loc_acc(gen_map, gt_map)
        metrics['Loc-ACC'] = acc
        metrics['hits'] = hits
        
        acck = compute_acck(gen_map, gt_map, ks=(1, 5, 10))
        metrics.update(acck)

        sd_series, si_series, dard_dist, stvd_dist = [], [], None, None
        for d in dates:
            route_str = gen_map.get(d)
            sd_series.extend(step_distance_series(route_str, id2geo))
            si_series.extend(step_interval_series(route_str))
            if vec:
                dist = dard_distribution(route_str, vec)
                dard_dist = dist if dard_dist is None else dard_dist + dist
            s_dist = stvd_distribution(route_str, id2geo)
            stvd_dist = s_dist if stvd_dist is None else stvd_dist + s_dist
        
        metrics['sd_series'] = sd_series
        metrics['si_series'] = si_series
        metrics['dard_dist'] = dard_dist
        metrics['stvd_dist'] = stvd_dist
        metrics['total_locs'] = total_locs
        all_metrics[name] = metrics

    # Calculate GT metrics once
    gt_sd_series, gt_si_series, gt_dard_dist, gt_stvd_dist = [], [], None, None
    for d in dates:
        route_str = gt_map.get(d)
        gt_sd_series.extend(step_distance_series(route_str, id2geo))
        gt_si_series.extend(step_interval_series(route_str))
        if vec:
            dist = dard_distribution(route_str, vec)
            gt_dard_dist = dist if gt_dard_dist is None else gt_dard_dist + dist
        s_dist = stvd_distribution(route_str, id2geo)
        gt_stvd_dist = s_dist if gt_stvd_dist is None else gt_stvd_dist + s_dist

    # Calculate JSD vs GT
    all_sds = gt_sd_series + [s for name in all_metrics for s in all_metrics[name]['sd_series']]
    all_sis = gt_si_series + [s for name in all_metrics for s in all_metrics[name]['si_series']]
    sd_bins = np.linspace(0.0, max(all_sds + [1.0]), 21)
    si_bins = np.linspace(0.0, max(all_sis + [60.0]), 25)
    gt_sd_p = hist_prob(gt_sd_series, sd_bins)
    gt_si_p = hist_prob(gt_si_series, si_bins)

    for name, metrics in all_metrics.items():
        sd_p = hist_prob(metrics['sd_series'], sd_bins)
        si_p = hist_prob(metrics['si_series'], si_bins)
        metrics['JSD_SD'] = jsd(sd_p, gt_sd_p)
        metrics['JSD_SI'] = jsd(si_p, gt_si_p)
        if vec and metrics['dard_dist'] is not None and gt_dard_dist is not None:
            metrics['JSD_DARD'] = jsd(metrics['dard_dist'], gt_dard_dist)
        if metrics['stvd_dist'] is not None and gt_stvd_dist is not None:
            metrics['JSD_STVD'] = jsd(metrics['stvd_dist'], gt_stvd_dist)

    # --- Print Report ---
    sorted_names = sorted(all_metrics.keys())
    header = f"| {'Metric':<12} |" + "".join([f" {name:<15} |" for name in sorted_names])
    separator = "-" * len(header)
    
    print("\n" + "="*len(header))
    print(f"Trajectory Generation Evaluation Report")
    print(f"Agent ID: {args.id}, Dataset: {args.dataset}, ExpDir: {args.exp_dir}")
    print("="*len(header))
    
    print(header)
    print(separator)
    
    total_locs = all_metrics[sorted_names[0]]['total_locs']
    row = f"| {'Loc-ACC':<12} |"
    for name in sorted_names:
        val = all_metrics[name].get('Loc-ACC', 0.0)
        hits = all_metrics[name].get('hits', 0)
        row += f" {val:.4f} ({hits:>{len(str(total_locs))}}/{total_locs}) |"
    print(row)

    for k in [1, 5, 10]:
        metric_name = f"Acc@{k}"
        row = f"| {metric_name:<12} |"
        for name in sorted_names:
            val = all_metrics[name].get(metric_name, 0.0)
            row += f" {val:<15.4f} |"
        print(row)
        
    print(separator)
    
    for metric in ['JSD_SD', 'JSD_SI', 'JSD_DARD', 'JSD_STVD']:
        row = f"| {metric:<12} |"
        for name in sorted_names:
            val = all_metrics[name].get(metric)
            row += f" {val:<15.4f} |" if val is not None else f" {'N/A':<15} |"
        print(row)
        
    print(separator)
    
    gt_sd_mean, gt_sd_median = (np.mean(gt_sd_series), np.median(gt_sd_series)) if gt_sd_series else (0,0)
    gt_si_mean, gt_si_median = (np.mean(gt_si_series), np.median(gt_si_series)) if gt_si_series else (0,0)

    print(f"| {'SD mean (GT)':<12} | {f'{gt_sd_mean:.4f}':<15} |")
    print(f"| {'SD median (GT)':<12} | {f'{gt_sd_median:.4f}':<15} |")
    print(f"| {'SI mean (GT)':<12} | {f'{gt_si_mean:.4f}':<15} |")
    print(f"| {'SI median (GT)':<12} | {f'{gt_si_median:.4f}':<15} |")

    print("="*len(header) + "\n")

if __name__ == '__main__':
    main()