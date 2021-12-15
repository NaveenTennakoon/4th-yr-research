import torch
import os
from .model import Model

def load_model(use_sfl=False):
    model = Model(
        vocab_size=41,
        dim=512,
        max_num_states=2,
        use_sfl=use_sfl,
    )

    while True:
        path = "./model/30.pth"
        try:
            model.load_state_dict(torch.load(path, "cpu"))
            break
        except Exception as e:
            print(e)
            print("Fail to load model, re-downloading ...")
            os.remove(path)
    return model