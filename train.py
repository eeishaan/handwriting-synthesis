import os
import pathlib
import time

import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from tqdm import tqdm

from constants import BATCH_FIRST
from data import get_loader
from models.prediction import PredModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train():
    batch_size = 128
    lr = 1e-4
    model = PredModel(2, 3, 20, batch_size=batch_size).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loader = get_loader("data/strokes-py2.npy", batch_size)
    num_epochs = 100
    # bce_loss = torch.nn.BCELoss(reduction="mean")
    save_dir = pathlib.Path(f"runs/{batch_size}_{lr}_{time.time()}")
    os.makedirs(save_dir)
    for epoch in tqdm(range(num_epochs)):
        for x, labels, mask in loader:
            x = x.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            out = model(x)
            model.reset()
            prob_t, e_t = model.infer(out, x, mask)  # shape = (b,s)

            # calculate loss
            loss = (
                -prob_t
                + binary_cross_entropy_with_logits(
                    input=e_t, target=labels, weight=mask, reduction="sum"
                )
                / mask.sum()
            )

            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"Epoch {epoch}: Loss {loss.detach()}")
        # checkpoint weights
        torch.save(
            {"model": model.state_dict(), "optim": optim.state_dict(), "epoch": epoch},
            open(save_dir / "model.pt", "wb"),
        )


if __name__ == "__main__":
    train()
