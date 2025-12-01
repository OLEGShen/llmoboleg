"""
Person class to store necessary information about a person in the simulation.
"""

from engine.utilities.retrieval_helper import *
from engine.experimental.wrapper import build_vimn
from engine.experimental.intent_retriever import IntentRetriever
from engine.experimental.vimn import DataVectorizer
from engine.vimn_core import VIMN
from engine.neuro_bridge import IntentionTranslator
from engine.memory_manager import DualMemory
import os
import torch


class Person:
    def __init__(self, name, model, person_id=-10, fast=False):
        self.retriever = None
        self.train_routine_list = None  # list of training routines
        self.test_routine_list = None  # list of testing routines
        self.name = name
        self.city_area = None
        self.llm = model
        self.cat = None
        self.id = person_id
        self.domain_knowledge = None
        self.neg_routines = None
        self.attribute = None
        self.loc_cat = None
        self.top_k_routine = 6
        self.fast = fast
        self.vec = None
        self.vimn = None
        self.dual_memory = None
        self.intent_translator = None
        print("Person {} is created".format(self.name))

    def init_retriever(self, ):
        self.retriever = TemporalRetriever(self.train_routine_list,
                                           6,
                                           is_train=1, class_id_map=self.loc_cat)

    def init_intent_retriever(self, ckpt_path: str = './engine/experimental/checkpoints/vimn_lite.pt'):
        vec, model = build_vimn('./data/loc_map.pkl', './data/location_activity_map.pkl', 128, 256)
        try:
            import torch
            state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state['model'])
            from engine.experimental.vimn import IntentContrastiveHead
            head = IntentContrastiveHead(256)
            head.load_state_dict(state['head'])
        except Exception:
            from engine.experimental.vimn import IntentContrastiveHead
            head = IntentContrastiveHead(256)
        self.intent_retriever = IntentRetriever(vec, model, head, self.train_routine_list, top_k=6,
                                                test_trajs=self.test_routine_list,
                                                time_window=2,
                                                weekend_only=None)

    def init_neuro_symbolic(self, embed_dim: int = 128, hidden_dim: int = 256, allowed_poi_ids=None, allowed_act_names=None):
        self.vec = DataVectorizer('./data/loc_map.pkl', './data/location_activity_map.pkl',
                                  allowed_poi_ids=allowed_poi_ids,
                                  allowed_act_names=allowed_act_names)
        self.vimn = VIMN(num_pois=len(self.vec.poi_vocab), num_acts=len(self.vec.act_vocab),
                         embed_dim=embed_dim, hidden_dim=hidden_dim)
        id2name = [None] * len(self.vec.act_vocab)
        for name, idx in self.vec.act_vocab.items():
            if idx < len(id2name):
                id2name[idx] = str(name)
        self.intent_translator = IntentionTranslator(id2name)
        self.dual_memory = DualMemory(self.vimn, self.vec, self.vec.act_vocab)

    def get_vimn_hint_text(self, date_: str, demo: str = None):
        if self.dual_memory is None:
            self.init_neuro_symbolic()
        try:
            if hasattr(self, 'intent_retriever') and self.intent_retriever is not None:
                top = self.intent_retriever.retrieve(date_ if demo is None else demo)
                cats = []
                for r in top:
                    items = r.split(': ')[-1].replace(', ', ' at ').split(' at ')[::2]
                    for it in items:
                        name = it.split('#')[0].strip()
                        c = self.loc_cat.get(name)
                        if c is not None:
                            cats.append(c)
                from collections import Counter
                cnt = Counter(cats)
                top3 = [k for k, _ in cnt.most_common(3)]
                if len(top3) > 0:
                    return f"我的习惯记忆通常在此时会去：[{', '.join(map(str, top3))}]。"
        except Exception:
            pass
        h = self.dual_memory.get_state()
        if h is None and self.train_routine_list:
            last = self.train_routine_list[-1]
            try:
                poi_ids, act_ids, time_ids = self.vec.vectorize_sequence([last])
                h = self.vimn.forward(poi_ids, act_ids, time_ids)
                self.dual_memory.h = h
            except Exception:
                pass
        if h is None:
            import torch
            h = torch.zeros((1, self.vimn.gru.hidden_size))
        logits = self.vimn.translate(h)
        return self.intent_translator.get_natural_language_hint(logits)

    def ingest_generated_step(self, loc_with_id: str, time_str: str):
        if self.dual_memory is None:
            self.init_neuro_symbolic()
        try:
            poi_id = int(str(loc_with_id).split('#')[-1])
        except Exception:
            poi_id = self.vec.poi_vocab.get('UNK')
        act_name = self.vec.id2act.get(poi_id, 'UNK')
        try:
            hour = int(time_str.split(':')[0])
        except Exception:
            hour = 0
        self.dual_memory.ingest_location(poi_id, act_name, hour)
