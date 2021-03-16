import enum
import os
import pathlib
import shutil
import time

import sys
import torch
from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from constants import BATCH_FIRST
from data import get_loader
from models.prediction import PredModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root = "/tmp/" if torch.cuda.is_available() else ""


def train(second_stage: bool = False):
    batch_size = 64
    lr = 1e-4
    with_texts = True if second_stage else False
    model = PredModel(
        3, 3, 20, batch_size=batch_size, hidden_dim=400, with_texts=with_texts
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # optim = torch.optim.RMSprop(
    #     model.parameters(), lr=lr, alpha=0.9, momentum=0.95, eps=1e-4
    # )
    loader, dataset = get_loader(
        root + "data/strokes-py3.npy", batch_size, with_texts=with_texts
    )
    # loader, dataset = get_loader("data/strokes-py3.npy", batch_size)
    num_epochs = 100
    save_dir = f"{root}runs/{batch_size}_{lr}_{time.time()}"
    os.makedirs(save_dir)
    bptt_steps = 100
    # model.generate(device)

    def _single_step(x, labels, label_mask, input_mask, seqs=None):
        x = x.to(device)
        labels = labels[:, 1:].to(device)
        label_mask = label_mask[:, 1:].to(device)
        input_mask = input_mask[:, :-1].to(device)
        if second_stage:
            seqs = seqs.to(device)
        out = model(x[:, :-1, :], seqs)

        out.retain_grad()

        # with torch.autograd.detect_anomaly():
        prob_t, e_t, sse = model.infer(
            out, x[:, 1:, 1:], input_mask, label_mask
        )  # shape = (b,s)

        # calculate loss
        prob_loss = -prob_t
        e_t_loss = (
            binary_cross_entropy(
                input=e_t[input_mask], target=labels[label_mask], reduction="none"
            )
        ).sum() / len(label_mask.sum(1) > 1)

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

        for batch_val in loader:

            model.reset()
            if not with_texts:
                all_x, all_labels, all_label_mask, all_input_mask = batch_val
                xs = torch.split(all_x, bptt_steps, 1)
                ls = torch.split(all_labels, bptt_steps, 1)
                i_ms = torch.split(all_input_mask, bptt_steps, 1)
                l_ms = torch.split(all_label_mask, bptt_steps, 1)
                for i, (x, labels, input_mask, label_mask) in enumerate(
                    zip(xs, ls, i_ms, l_ms)
                ):
                    # for x, labels, label_mask, input_mask in loader:
                    prob_loss, e_t_loss, sse = _single_step(
                        x, labels, label_mask, input_mask
                    )
            else:
                all_x, all_labels, all_label_mask, all_input_mask, all_seqs = batch_val
                prob_loss, e_t_loss, sse = _single_step(
                    all_x, all_labels, all_label_mask, all_input_mask, all_seqs
                )
                e_loss += e_t_loss
                p_loss += prob_loss
                epoch_sse += sse

        len_batches = len(loader)
        if not with_texts:
            len_batches *= len(xs)
        e_loss /= len_batches
        p_loss /= len_batches
        epoch_loss = e_loss + p_loss
        print(
            f"Epoch {epoch}: Loss {epoch_loss}({e_loss}+{p_loss}) | SSE: {epoch_sse / len_batches}"
        )
        # checkpoint weights
        torch.save(
            {"model": model.state_dict(), "optim": optim.state_dict(), "epoch": epoch},
            open(save_dir + "/model.pt", "wb"),
        )

    shutil.move(save_dir, save_dir[len(root) :])


if __name__ == "__main__":
    train(eval(sys.argv[1]))
