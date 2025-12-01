import os
import torch
from engine.vimn_core import VIMN
from engine.experimental.vimn import DataVectorizer


def load_vimn_gru_ckpt(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    state = torch.load(ckpt_path, map_location='cpu')
    meta = state.get('meta', {})
    allowed_poi = meta.get('allowed_poi_ids')
    allowed_act = meta.get('allowed_act_names')
    embed_dim = int(meta.get('embed_dim', 128))
    hidden_dim = int(meta.get('hidden_dim', 256))
    vec = DataVectorizer('./data/loc_map.pkl', './data/location_activity_map.pkl',
                         allowed_poi_ids=allowed_poi, allowed_act_names=allowed_act)
    model = VIMN(len(vec.poi_vocab), len(vec.act_vocab), embed_dim, hidden_dim)
    sd = state.get('model', state)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return vec, model, meta