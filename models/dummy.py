import os

import torch
from data import get_embedding
from utils.helper import inverse_transform

from models.prediction import PredModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
first_stage = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "asset", "first_stage.pt"
)
second_stage = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "asset", "second_stage.pt"
)


def generate_unconditionally(random_seed=1):
    # Input:
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    state = torch.load(open(first_stage, "rb"), map_location=device)["model"]
    model = PredModel(3, 1, 20, batch_size=1, hidden_dim=900, with_texts=False).to(
        device
    )
    model.load_state_dict(state)
    model.eval()
    x = model.generate(random_seed, device, bias=0.2)
    x = torch.cat(x, axis=0).squeeze()
    inv = inverse_transform(x[:, 1:])
    x = torch.cat([x[:, 0:1], inv], axis=1)
    x[-1, 0] = 1
    return x.numpy()


def generate_conditionally(text="welcome to lyrebird", random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)

    state = torch.load(open(second_stage, "rb"), map_location=device)["model"]
    model = PredModel(3, 1, 20, batch_size=1, hidden_dim=900, with_texts=True).to(
        device
    )
    model.load_state_dict(state)
    model.eval()
    seq = torch.from_numpy(get_embedding(text)).unsqueeze(0).float()

    x = model.generate_with_seq(random_seed, device, seqs=seq, bias=2)
    x = torch.cat(x, axis=0).squeeze()
    inv = inverse_transform(x[:, 1:])
    x = torch.cat([x[:, 0:1], inv], axis=1)
    x[-1, 0] = 1
    return x.numpy()


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return "welcome to lyrebird"
