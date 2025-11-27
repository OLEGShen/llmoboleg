import os
import sys
import json
import pickle
import argparse
import subprocess


def run_cmd(cmd):
    print('RUN:', cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = p.communicate()
    code = p.returncode
    print(out.decode(errors='ignore'))
    if code != 0:
        raise RuntimeError(f'Command failed: {cmd}')


def read_results(result_dir):
    path = os.path.join(result_dir, 'results.pkl')
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as f:
        return pickle.load(f)


def parse_hour(route: str):
    try:
        hour = int(route.split(' at ')[-1][:2])
        return hour
    except Exception:
        return None


def load_truth(result_dir):
    path = os.path.join(result_dir, 'results.pkl')
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as f:
        return pickle.load(f)


def eval_pair(gen_map, gt_map, loc_cat):
    # gen_map: {date: "Activities at DATE: ..."}
    # gt_map: {date: "Activities at DATE: ..."}
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

    cat_hits = 0
    time_hits = 0
    json_ok = 0
    for d in dates:
        g = gen_map.get(d)
        r = gt_map.get(d)
        if g is None or r is None:
            continue
        # 类别一致性：按出现次序比较类别是否相等（非空）
        gc = route_to_cats(g)
        rc = route_to_cats(r)
        m = min(len(gc), len(rc))
        cat_hits += sum(1 for i in range(m) if gc[i] is not None and rc[i] is not None and gc[i] == rc[i])
        # 时段匹配：比较首个时间点是否在±2小时内
        gh = parse_hour(g)
        rh = parse_hour(r)
        if gh is not None and rh is not None and abs(gh - rh) <= 2:
            time_hits += 1
        # JSON 成功率：若生成文本能被解析为 plan（在 pipeline 内部成功写入 results.pkl 视为成功）
        json_ok += 1

    return {
        'count': len(dates),
        'cat_match': cat_hits / max(1, len(dates)),
        'time_match': time_hits / max(1, len(dates)),
        'json_ok': json_ok / max(1, len(dates)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='2019')
    ap.add_argument('--ids', type=str, required=True, help='逗号分隔的id列表')
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--intent_ckpt', type=str, default='./engine/experimental/checkpoints/vimn_lite.pt')
    args = ap.parse_args()

    ids = [int(s) for s in args.ids.split(',') if s.strip()]
    scenario_tag = {'2019': 'normal', '2021': 'abnormal', '20192021': 'normal_abnormal'}[args.dataset]

    report = {'baseline': {}, 'vimn': {}}

    for pid in ids:
        # baseline
        cmd_base = f"python generate.py --dataset {args.dataset} --mode 0 --id {pid} {'--fast' if args.fast else ''}"
        run_cmd(cmd_base)
        gen_dir = f"./result/{scenario_tag}/generated/llm_l/{pid}/"
        gt_dir = f"./result/{scenario_tag}/ground_truth/llm_l/{pid}/"
        gen_map = read_results(gen_dir)
        gt_map = load_truth(gt_dir)
        # 加载 loc_cat 以便类别映射
        with open(f"./data/{args.dataset}/{pid}.pkl", 'rb') as f:
            att = pickle.load(f)
        loc_cat = att[11] if len(att) > 11 else {}
        m_base = eval_pair(gen_map, gt_map, loc_cat)
        report['baseline'][pid] = m_base

        # vimn
        cmd_v = f"python generate.py --dataset {args.dataset} --mode 0 --id {pid} {'--fast' if args.fast else ''} --use_intent --intent_ckpt {args.intent_ckpt}"
        run_cmd(cmd_v)
        gen_map_v = read_results(gen_dir)
        m_v = eval_pair(gen_map_v, gt_map, loc_cat)
        report['vimn'][pid] = m_v

    # 汇总
    def avg(metrics_list, key):
        vals = [m.get(key, 0.0) for m in metrics_list if m]
        return sum(vals) / max(1, len(vals))

    baseline_list = list(report['baseline'].values())
    vimn_list = list(report['vimn'].values())
    summary = {
        'baseline': {
            'cat_match': avg(baseline_list, 'cat_match'),
            'time_match': avg(baseline_list, 'time_match'),
            'json_ok': avg(baseline_list, 'json_ok'),
        },
        'vimn': {
            'cat_match': avg(vimn_list, 'cat_match'),
            'time_match': avg(vimn_list, 'time_match'),
            'json_ok': avg(vimn_list, 'json_ok'),
        }
    }

    os.makedirs('./result/experiment', exist_ok=True)
    with open('./result/experiment/intent_vs_baseline.json', 'w') as f:
        json.dump({'report': report, 'summary': summary}, f, ensure_ascii=False, indent=2)
    print('Saved ./result/experiment/intent_vs_baseline.json')


if __name__ == '__main__':
    main()