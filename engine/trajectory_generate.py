from engine.prompt_template.prompt_paths import *
from engine.utilities.process_tools import *
from engine.llm_configs.gpt_structure import *
from engine.utilities.retrieval_helper import *
from engine.neuro_bridge import IntentionTranslator
from engine.vimn_core import VIMN
from engine.experimental.vimn import DataVectorizer
from engine.memory_manager import DualMemory
from engine.vimn_loader import load_vimn_gru_ckpt
import json
import os
import pickle
import torch


def mob_gen(person, mode=0, scenario_tag="normal", fast=False, use_vimn=True):
    infer_template = "./engine/prompt_template/one-shot_infer_mot.txt"
    # mode = 0 for learning based retrieval, 1 for evolving based retrieval
    describe_mot_template = "./engine/" + motivation_infer_prompt_paths[mode]
    motivation_ways = ["Following are the motivation that you want to achieve:",
                       "Following are the thing you focus in the last few days:"
                       ]
    mode_name = {0: "llm_l", 1: "llm_e"}
    generation_path = f"./result/{scenario_tag}/generated/{mode_name[mode]}/{str(person.id)}/"
    ground_truth_path = f"./result/{scenario_tag}/ground_truth/{mode_name[mode]}/{str(person.id)}/"
    if os.path.exists(generation_path) is False:
        os.makedirs(generation_path)
    if os.path.exists(ground_truth_path) is False:
        os.makedirs(ground_truth_path)

    results = {}
    reals = {}
    his_routine = person.train_routine_list[-person.top_k_routine:]
    test_iter = person.test_routine_list[:1] if fast else person.test_routine_list[:]
    try:
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
        if mode == 0:
            # learning based retrieved
            retrieve_route = person.retriever.retrieve(date_)
            demo = retrieve_route[0]
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
        if motivation is not None:
            prior = f"[Internal Intuition] {vimn_hint}"
            curr_input = [person.attribute, motivation, date_, ',  '.join(area), weekday, demo,
                          motivation_ways[mode],
                          prior]
        prompt = generate_prompt(curr_input, infer_template)
        max_trial = 3 if fast else 10
        trial = 0
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
        if mode == 0:
            person.retriever.nodes.append(reals[date_])
    # dump pkl
    with open(generation_path + "results.pkl", "wb") as f:
        pickle.dump(results, f)
    with open(ground_truth_path + "results.pkl", "wb") as f:
        pickle.dump(reals, f)
    print(generation_path)
    print(ground_truth_path)
