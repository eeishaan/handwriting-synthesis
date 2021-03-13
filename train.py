import enum
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
from torch.nn.utils import clip_grad_norm_


def train():
    batch_size = 64
    lr = 1e-4
    model = PredModel(2, 3, 20, batch_size=batch_size, hidden_dim=400).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # optim = torch.optim.RMSprop(
    #     model.parameters(), lr=lr, alpha=0.9, momentum=0.95, eps=1e-4
    # )
    loader, dataset = get_loader("/tmp/data/strokes-py3.npy", batch_size)
    # loader, dataset = get_loader("data/strokes-py3.npy", batch_size)
    num_epochs = 200
    save_dir = pathlib.Path(f"runs/{batch_size}_{lr}_{time.time()}")
    os.makedirs(save_dir)
    bptt_steps = 100

    def _single_epoch(x, labels, label_mask, input_mask):
        x = x.to(device)
        labels = labels.to(device)
        label_mask = label_mask.to(device)
        input_mask = input_mask.to(device)

        out = model(x)
        out.retain_grad()

        # with torch.autograd.detect_anomaly():
        prob_t, e_t, sse = model.infer(out, x, input_mask, label_mask)  # shape = (b,s)

        # calculate loss
        prob_loss = -prob_t
        e_t_loss = (
            binary_cross_entropy_with_logits(
                input=e_t[input_mask], target=labels[label_mask], reduction="none"
            )
        ).sum() / len(label_mask.sum(0) > 1)

        loss = prob_loss + e_t_loss

        optim.zero_grad()
        loss.backward()

        # clip w grad and output grad
        clip_grad_norm_(model.parameters(), 10)
        out.grad.clamp_(-100, 100)

        optim.step()

        return prob_loss.detach(), e_t_loss.detach(), sse

    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = 0
        e_loss = 0
        p_loss = 0
        epoch_sse = 0

        for x, labels, label_mask, input_mask in loader:

            #     xs = torch.split(all_x, bptt_steps, 1)
            #     ls = torch.split(all_labels, bptt_steps, 1)
            #     ms = torch.split(all_mask, bptt_steps, 1)
            #     batch_loss = 0
            # for i, (x, labels, mask) in enumerate(zip(xs, ls, ms)):

            prob_loss, e_t_loss, sse = _single_epoch(x, labels, label_mask, input_mask)
            e_loss += e_t_loss
            p_loss += prob_loss
            epoch_sse += sse

        len_batches = len(loader) + 1
        e_loss /= len_batches
        p_loss /= len_batches
        epoch_loss = e_loss + p_loss
        print(
            f"Epoch {epoch}: Loss {epoch_loss}({e_loss}+{p_loss}) | SSE: {epoch_sse / len_batches}"
        )
        # checkpoint weights
        torch.save(
            {"model": model.state_dict(), "optim": optim.state_dict(), "epoch": epoch},
            open(save_dir / "model.pt", "wb"),
        )


if __name__ == "__main__":
    train()
