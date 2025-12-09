from engine.prompt_template.prompt_paths import *
from engine.utilities.process_tools import *
from engine.llm_configs.gpt_structure import *
from engine.utilities.retrieval_helper import *
from engine.neuro_bridge import IntentionTranslator
from engine.vimn_core import VIMN
from engine.experimental.vimn import DataVectorizer
from engine.memory_manager import DualMemory
from engine.vimn_loader import load_vimn_gru_ckpt
from engine.memento_policy import MementoPolicyNet
from engine.dpo_gating import DPOGatingNet
import json
import os
import pickle
import torch


def mob_gen(person, mode=0, scenario_tag="normal", fast=False, use_vimn=True, use_memento=False, use_gating=False, gating_ckpt=None, vimn_ckpt=None, memento_ckpt=None):
    infer_template = "./engine/prompt_template/one-shot_infer_mot.txt"
    # mode = 0 for learning based retrieval, 1 for evolving based retrieval
    describe_mot_template = "./engine/" + motivation_infer_prompt_paths[mode]
    motivation_ways = ["Following are the motivation that you want to achieve:",
                       "Following are the thing you focus in the last few days:"
                       ]
    mode_name = {0: "llm_l", 1: "llm_e"}
    variant = []
    if use_vimn:
        variant.append("vimn")
    if use_memento:
        variant.append("memento")
    if use_gating:
        variant.append("gating")
    variant_dir = "none" if len(variant) == 0 else "_".join(variant)
    generation_path = f"./result/{scenario_tag}/generated/{mode_name[mode]}/{str(person.id)}/{variant_dir}/"
    ground_truth_path = f"./result/{scenario_tag}/ground_truth/{mode_name[mode]}/{str(person.id)}/"
    if os.path.exists(generation_path) is False:
        os.makedirs(generation_path)
    if os.path.exists(ground_truth_path) is False:
        os.makedirs(ground_truth_path)

    results = {}
    reals = {}
    details = {}
    his_routine = person.train_routine_list[-person.top_k_routine:]
    test_iter = person.test_routine_list[:1] if fast else person.test_routine_list[:]
    try:
        if isinstance(vimn_ckpt, str) and os.path.exists(vimn_ckpt):
            ckpt_global = vimn_ckpt
        else:
            if scenario_tag == 'normal':
                ckpt_global = './engine/experimental/checkpoints/vimn_global_gru_2019.pt'
            elif scenario_tag == 'abnormal':
                ckpt_global = './engine/experimental/checkpoints/vimn_global_gru_2021.pt'
            else:
                ckpt_global = './engine/experimental/checkpoints/vimn_global_gru_20192021.pt'
        ckpt_id = f"./engine/experimental/checkpoints/batch/vimn_best_gru_{'2019' if scenario_tag=='normal' else '2021'}_{person.id}.pt"
        vec, vimn, meta = None, None, None
        if os.path.exists(ckpt_global):
            vec, vimn, meta = load_vimn_gru_ckpt(ckpt_global)
            u2i = meta.get('user2idx', {})
            user_idx = u2i.get(str(person.id))
        elif os.path.exists(ckpt_id):
            vec, vimn, meta = load_vimn_gru_ckpt(ckpt_id)
            user_idx = None
        else:
            vec, vimn, meta, user_idx = None, None, None, None
        id2name = [None] * len(vec.act_vocab)
        for name, idx in vec.act_vocab.items():
            id2name[idx] = str(name)
        translator = IntentionTranslator(id2name)
        dm = DualMemory(vimn, vec, vec.act_vocab, user_idx=user_idx)
        try:
            person.init_neuro_symbolic(allowed_poi_ids=meta.get('allowed_poi_ids'), allowed_act_names=meta.get('allowed_act_names'), user_idx=user_idx, pre_vec=vec, pre_vimn=vimn)
        except Exception:
            pass
    except Exception:
        vec = None
        vimn = None
        translator = None
        dm = None
    for test_route in test_iter:
        date_ = test_route.split(": ")[0].split(" ")[-1]
        # get motivation
        consecutive_past_days = check_consecutive_dates(his_routine, date_)
        memento_strength = 0.0
        if mode == 0:
            # learning based retrieved
            retrieve_route = person.retriever.retrieve(date_)
            demo = retrieve_route[0]
            if use_memento:
                try:
                    memento_path = memento_ckpt if (isinstance(memento_ckpt, str) and os.path.exists(memento_ckpt)) else './engine/experimental/checkpoints/memento_policy.pt'
                    if os.path.exists(memento_path):
                        st = torch.load(memento_path, map_location='cpu')
                        input_dim = st.get('input_dim')
                        hidden_dim = st.get('hidden_dim', 512)
                        mp = MementoPolicyNet(input_dim=input_dim if input_dim is not None else (int(1440/60)*len(set(person.loc_cat.values()))*2), hidden_dim=hidden_dim)
                        mp.load_state_dict(st['state_dict'])
                        mp.eval()
                        act_map_local = _build_act_map(person.loc_cat)
                        state_vec = _route_vec(demo, person.loc_cat, act_map_local)
                        cands = list(retrieve_route)
                        try:
                            if hasattr(person, 'intent_retriever') and person.intent_retriever is not None:
                                intent_top = person.intent_retriever.retrieve(date_)
                                if isinstance(intent_top, list) and len(intent_top) > 0:
                                    cands = list(dict.fromkeys(list(retrieve_route) + intent_top))
                        except Exception:
                            pass
                        if len(cands) > 0:
                            se = state_vec.unsqueeze(0).expand(len(cands), -1)
                            me_list = []
                            for r in cands:
                                v = _route_vec(r, person.loc_cat, act_map_local)
                                me_list.append(v)
                            me = torch.stack(me_list, dim=0)
                            x = torch.cat([se, me], dim=-1)
                            with torch.no_grad():
                                scores = mp(x)
                            try:
                                memento_strength = float(torch.softmax(scores.reshape(-1), dim=0).max().item())
                            except Exception:
                                memento_strength = 0.0
                            idx = int(torch.argmax(scores).item())
                            demo = cands[idx]
                except Exception:
                    pass
            try:
                if hasattr(person, 'intent_retriever') and person.intent_retriever is not None:
                    intent_top = person.intent_retriever.retrieve(date_)
                    if len(intent_top) > 0:
                        demo = intent_top[0]
            except Exception:
                pass
        else:
            # evolving based retrieved
            demo = his_routine[-1]

        hint = ""
        curr_input = [person.attribute, "Go to " + demo.split(": ")[-1], consecutive_past_days, hint]

        prompt = generate_prompt(curr_input, describe_mot_template)
        area = retrieve_loc(person, demo)
        motivation = execute_prompt(prompt, person.llm, objective=f"Think about motivation")
        motivation = first2second(motivation)
        his_routine = his_routine[1:] + [test_route]
        weekday = find_detail_weekday(date_)
        vimn_hint = ""
        if use_vimn:
            try:
                vimn_hint = person.get_vimn_hint_text(date_, demo)
            except Exception:
                vimn_hint = ""
        vimn_topk = []
        if use_vimn:
            try:
                vimn_topk = person.get_vimn_topk(date_, demo, top_k=10)
            except Exception:
                vimn_topk = []
        if motivation is not None:
            def _entropy_from_topk(_topk):
                import math
                if not isinstance(_topk, list) or len(_topk) == 0:
                    return 1.0
                _ps = [float(d.get('prob', 0.0)) for d in _topk]
                _s = sum(_ps)
                if _s <= 0:
                    return 1.0
                _p = [x/_s for x in _ps]
                _H = sum([-pi*math.log(max(1e-8, pi)) for pi in _p])
                _Hmax = math.log(len(_p))
                return float(_H / max(1e-8, _Hmax))
            prior = f"[Internal Intuition] {vimn_hint}"
            if use_gating:
                try:
                    import math, datetime as dt
                    _entropy = _entropy_from_topk(vimn_topk)
                    _mem_score = memento_strength
                    y_, m_, d_ = map(int, date_.split('-'))
                    _wd = dt.date(y_, m_, d_).weekday()
                    _emb = [1.0 if i == _wd else 0.0 for i in range(7)]
                    _hour = 0
                    _emb += [math.sin(2*math.pi*_hour/24.0), math.cos(2*math.pi*_hour/24.0)]
                    if len(_emb) < 32:
                        _emb += [0.0] * (32 - len(_emb))
                    _state = torch.tensor([_entropy, _mem_score] + _emb[:32], dtype=torch.float32)
                    _gate = DPOGatingNet()
                    if gating_ckpt and os.path.exists(gating_ckpt):
                        _sd = torch.load(gating_ckpt, map_location='cpu')
                        try:
                            _gate.load_state_dict(_sd)
                        except Exception:
                            pass
                    with torch.no_grad():
                        _logits = _gate.forward(_state).squeeze(0)
                        _a = int(torch.argmax(_logits).item())
                    if _a == 0:
                        prior = f"[Internal Intuition] {vimn_hint}"
                    elif _a == 1:
                        prior = f"[Experience Memory] {demo.split(': ')[-1]}"
                    else:
                        prior = f"[Internal Intuition] {vimn_hint}\n[Experience Memory] {demo.split(': ')[-1]}"
                except Exception:
                    prior = f"[Internal Intuition] {vimn_hint}"
            curr_input = [person.attribute, motivation, date_, ',  '.join(area), weekday, demo,
                          motivation_ways[mode],
                          prior]
        prompt = generate_prompt(curr_input, infer_template)
        max_trial = 3 if fast else 10
        trial = 0
        contents = ""
        while trial < max_trial:
            contents = execute_prompt(prompt, person.llm,
                                      objective=f"one_shot_infer_response_{len(results) + 1}/{len(person.test_routine_list)}_{trial}")
            try:
                res = json.loads(contents)
                valid_generation(person, f"Activities at {date_}: " + ', '.join(res["plan"]))
                try:
                    if dm is not None:
                        for item in res["plan"]:
                            if " at " in item:
                                loc, tim = item.split(" at ")
                                try:
                                    pid = int(loc.split('#')[-1])
                                except Exception:
                                    pid = vec.poi_vocab.get('UNK') if vec is not None else 0
                                try:
                                    hour = int(tim.split(':')[0])
                                except Exception:
                                    hour = 0
                                act = vec.id2act.get(pid, 'UNK') if vec is not None else 'UNK'
                                dm.ingest_location(pid, act, hour)
                except Exception:
                    pass
            except Exception as e:
                print(e)
                trial += 1
                continue
            break
        if trial >= max_trial:
            res = {"plan": demo.split(": ")[-1]}
        print(contents)
        print("Motivation: ", motivation)
        print("Real: ", test_route)
        reals[date_] = test_route
        results[date_] = f"Activities at {date_}: " + ', '.join(res["plan"])
        try:
            details[date_] = {
                "motivation": motivation,
                "vimn_hint": vimn_hint,
                "vimn_topk": vimn_topk,
                "area": area,
                "weekday": weekday,
                "demo": demo,
                "raw_contents": contents,
                "plan": res.get("plan", []),
                "memento_top1_score": memento_strength,
                "memento_strength": memento_strength,
                "top1_score": memento_strength
            }
        except Exception:
            pass
        if mode == 0:
            person.retriever.nodes.append(reals[date_])
    # dump pkl
    with open(generation_path + "results.pkl", "wb") as f:
        pickle.dump(results, f)
    with open(ground_truth_path + "results.pkl", "wb") as f:
        pickle.dump(reals, f)
    try:
        with open(generation_path + "details.pkl", "wb") as f:
            pickle.dump(details, f)
    except Exception:
        pass
    print(generation_path)
    print(ground_truth_path)
def _extract_items(route: str):
    part = route.split(": ", 1)[1]
    segs = [s.strip() for s in part.split(',')]
    items = []
    import re
    for seg in segs:
        m_loc = re.search(r"(.+?)#(\d+)", seg)
        m_tm = re.search(r"(\d{2}):(\d{2})", seg)
        if m_loc and m_tm:
            loc_name = m_loc.group(1).strip()
            pid = int(m_loc.group(2))
            hh = int(m_tm.group(1)); mm = int(m_tm.group(2))
            items.append((loc_name, pid, hh*60+mm))
    return items

def _build_act_map(loc_cat: dict):
    acts = sorted(set(loc_cat.values()))
    return {a: i for i, a in enumerate(acts)}

def _route_vec(route: str, loc_cat: dict, act_map: dict, interval: int = 60):
    bins = int(1440 / interval)
    import torch as _t
    mat = _t.zeros((bins, len(act_map)), dtype=_t.float32)
    items = _extract_items(route)
    for loc_name, _, tmin in items:
        cat = loc_cat.get(loc_name)
        if cat is None or cat not in act_map:
            continue
        b = max(0, min(bins-1, tmin // interval))
        mat[b, act_map[cat]] += 1.0
    return mat.reshape(-1)
