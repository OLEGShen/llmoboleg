"""
Person class to store necessary information about a person in the simulation.
"""

from engine.utilities.retrieval_helper import *
from engine.experimental.wrapper import build_vimn
from engine.experimental.intent_retriever import IntentRetriever


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
