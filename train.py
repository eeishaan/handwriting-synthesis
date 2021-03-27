import os
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from constants import BATCH_FIRST
from data import get_loader
from models.prediction import PredModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root = ""  # "/tmp/" if torch.cuda.is_available() else ""


torch.manual_seed(0)
np.random.seed(0)


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    plt.figure(figsize=((10, 20)))
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )
    plt.savefig("grad.png")
    plt.close()


@torch.no_grad()
def surgery(old_model, new_model):
    old_lstm_layer = old_model.lstm.layers[0]
    new_model.lstm.layers[0].weight_ih[...] = old_lstm_layer.weight_ih_l0[...]
    new_model.lstm.layers[0].weight_hh[...] = old_lstm_layer.weight_hh_l0[...]
    new_model.lstm.layers[0].bias_ih[...] = old_lstm_layer.bias_ih_l0[...]
    new_model.lstm.layers[0].bias_hh[...] = old_lstm_layer.bias_hh_l0[...]
    new_model.lstm.input_proj[0].weight[...] = old_model.lstm.input_proj[0].weight[...]
    new_model.lstm.input_proj[0].bias[...] = old_model.lstm.input_proj[0].bias[...]
    new_model.lstm.final_projs[0].weight[...] = old_model.lstm.final_projs[0].weight[
        ...
    ]
    new_model.lstm.final_projs[0].bias[...] = old_model.lstm.final_projs[0].bias[...]
    new_model.lstm.bias[...] = old_model.lstm.bias[...]
    return new_model


def train(second_stage: bool = False):
    batch_size = 64
    lr = 1e-3
    with_texts = True if second_stage else False
    # if with_texts:
    #     new_model = PredModel(
    #         3, 1, 20, batch_size=batch_size, hidden_dim=900, with_texts=with_texts
    #     ).to(device)
    #     old_model = PredModel(
    #         3, 1, 20, batch_size=batch_size, hidden_dim=900, with_texts=False
    #     ).to(device)
    #     save_dir = "runs/64_0.001_1616506139.0575218" + "/model.pt"
    #     # save_dir = "beluga/64_0.001_1616506139.0575218.pt"
    #     state = torch.load(open(save_dir, "rb"), map_location=torch.device(device))
    #     old_model.load_state_dict(state["model"])
    #     model = surgery(old_model, new_model)
    #     del old_model
    model = PredModel(
        3, 1, 20, batch_size=batch_size, hidden_dim=900, with_texts=with_texts
    ).to(device)
    # model = (model)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    # save_dir = "runs/64_0.0001_1616797921.399419"
    save_dir = "runs/64_0.001_1616852698.2713563"
    state = torch.load(
        open(save_dir + "/model.pt", "rb"), map_location=torch.device(device)
    )
    model.load_state_dict(state["model"])
    optim.load_state_dict(state["optim"])
    start_epoch = state["epoch"]
    # start_epoch = 0
    # optim = torch.optim.RMSprop(
    #     model.parameters(), lr=lr, alpha=0.9, momentum=0.95, eps=1e-4
    # )
    loader, dataset = get_loader(
        root + "data/strokes-py3.npy", batch_size, with_texts=with_texts
    )
    # loader, dataset = get_loader("data/strokes-py3.npy", batch_size)
    num_epochs = 100
    # save_dir = f"{root}runs/{batch_size}_{lr}_{time.time()}"
    # os.makedirs(save_dir)
    bptt_steps = 100
    # model.generate_with_seq(1, device, torch.randn(1, 17, 58))

    is_scaling = False
    scaler = torch.cuda.amp.GradScaler(enabled=is_scaling)

    def _single_step(
        x, labels, label_mask, input_mask, seqs=None, seq_mask=None, alpha=0
    ):
        x = x.to(device)
        labels = x[:, :, 0]
        # labels = labels.to(device)
        label_mask = label_mask.to(device)
        input_mask = input_mask.to(device)
        if second_stage:
            seqs = seqs.to(device)
            seq_mask = seq_mask.to(device)
        with torch.cuda.amp.autocast(enabled=is_scaling):
            out = model(x, seqs, seq_mask=seq_mask)
            # out = out[:, :-1, :]
            out.retain_grad()

            # with torch.autograd.detect_anomaly():
            prob_t, e_t, sse = model.infer(
                out, x[:, :, 1:], input_mask, label_mask
            )  # shape = (b,s)

            # calculate loss
            prob_loss = -prob_t.sum()

            # one_mask = labels[label_mask] == 1
            # assert torch.isfinite(e_t).all()
            # assert torch.isfinite(labels).all()
            e_t_loss = binary_cross_entropy_with_logits(
                input=e_t[input_mask], target=labels[label_mask], reduction="none"
            )
            # e_t_loss[one_mask] *= dataset.w_1
            # e_t_loss[torch.logical_not(one_mask)] *= 1 - dataset.w_1

            e_t_loss = e_t_loss.sum()

            loss = (prob_loss + e_t_loss) / len(label_mask.sum(1) > 0)

        optim.zero_grad()
        # loss.backward()
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        # clip w grad and output grad
        out.grad.clamp_(-100, 100)
        clip_grad_norm_(model.parameters(), 10)
        # plot_grad_flow(model.named_parameters())

        # optim.step()
        scaler.step(optim)
        scaler.update()

        return prob_loss.detach(), e_t_loss.detach(), sse

    for epoch in tqdm(range(start_epoch, num_epochs)):
        model.train()
        epoch_loss = 0
        e_loss = 0
        p_loss = 0
        epoch_sse = 0

        for bi, batch_val in enumerate(loader):

            model.reset()
            # if not with_texts:
            # all_x, all_labels, all_label_mask, all_input_mask = batch_val
            # xs = torch.split(all_x, bptt_steps, 1)
            # ls = torch.split(all_labels, bptt_steps, 1)
            # i_ms = torch.split(all_input_mask, bptt_steps, 1)
            # l_ms = torch.split(all_label_mask, bptt_steps, 1)
            # for i, (x, labels, input_mask, label_mask) in enumerate(
            #     zip(xs, ls, i_ms, l_ms)
            # ):
            #     # for x, labels, label_mask, input_mask in loader:
            #     if x.shape[1] == 1:
            #         continue
            #     prob_loss, e_t_loss, sse = _single_step(
            #         x, labels, label_mask, input_mask
            #     )
            # else:
            if not with_texts:
                all_x, all_labels, all_label_mask, all_input_mask = batch_val
                seqs = None
                seq_mask = None
            else:
                (
                    all_x,
                    all_labels,
                    all_label_mask,
                    all_input_mask,
                    seqs,
                    seq_mask,
                ) = batch_val

            # try:
            prob_loss, e_t_loss, sse = _single_step(
                all_x,
                all_labels,
                all_label_mask,
                all_input_mask,
                seqs=seqs,
                seq_mask=seq_mask,
                alpha=epoch // num_epochs,
            )
            # except Exception as e:
            #     print(bi)
            #     raise e

            e_loss += e_t_loss / batch_size
            p_loss += prob_loss / batch_size
            epoch_sse += sse

        len_batches = len(loader)
        # if not with_texts:
        # len_batches *= len(xs)
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

    # shutil.move(save_dir, save_dir[len(root) :])


if __name__ == "__main__":
    train(eval(sys.argv[1]))
