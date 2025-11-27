import torch
from .vimn import DataVectorizer, VIMN_Lite


def build_vimn(loc_map_path: str = "./data/loc_map.pkl",
               act_map_path: str = "./data/location_activity_map.pkl",
               embed_dim: int = 64,
               hidden_dim: int = 128,
               allowed_poi_ids=None,
               allowed_act_names=None):
    vec = DataVectorizer(loc_map_path, act_map_path,
                         allowed_poi_ids=allowed_poi_ids,
                         allowed_act_names=allowed_act_names)
    model = VIMN_Lite(num_pois=len(vec.poi_vocab), num_acts=len(vec.act_vocab),
                      embed_dim=embed_dim, hidden_dim=hidden_dim)
    return vec, model


def encode_intent(vec: DataVectorizer, model: VIMN_Lite, trajectory_list):
    poi_ids, act_ids, time_ids = vec.vectorize_sequence(trajectory_list)
    with torch.no_grad():
        v = model(poi_ids, act_ids, time_ids)
    return v