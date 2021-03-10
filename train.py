import enum
import os
import pathlib
import time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

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
    # model.generate(device)
    bptt_steps = 100
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for all_x, all_labels, all_mask, all_lens in loader:
            xs = torch.split(all_x, bptt_steps, 1)
            ls = torch.split(all_labels, bptt_steps, 1)
            ms = torch.split(all_mask, bptt_steps, 1)
            for i, (x, labels, mask) in enumerate(zip(xs, ls, ms)):
                new_lens = torch.clamp(all_lens - bptt_steps * i, min=0, max=bptt_steps)
                x = pack_padded_sequence(
                    x, new_lens, enforce_sorted=False, batch_first=BATCH_FIRST
                )
                x = x.to(device)
                labels = labels[:, :-1].to(device)
                mask = mask[:, :-1].to(device)
                out = model(x)
                prob_t, e_t, sse = model.infer(out, x, mask)  # shape = (b,s)

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
            model.reset()

        print(f"Epoch {epoch}: Loss {loss.detach()} | SSE: {sse}")
        # checkpoint weights
        torch.save(
            {"model": model.state_dict(), "optim": optim.state_dict(), "epoch": epoch},
            open(save_dir / "model.pt", "wb"),
        )


if __name__ == "__main__":
    train()
