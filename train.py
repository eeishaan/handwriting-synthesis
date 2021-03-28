import argparse
import os
import time

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from constants import BATCH_FIRST
from data import get_loader
from models.prediction import PredModel
from utils.helper import surgery as surgery_fn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root = ""  # "/tmp/" if torch.cuda.is_available() else ""

torch.manual_seed(0)
np.random.seed(0)


def train(second_stage: bool = False, surgery: str = None, load_path: str = None):
    batch_size = 64
    lr = 1e-3
    model = PredModel(
        3, 1, 20, batch_size=batch_size, hidden_dim=900, with_texts=second_stage
    ).to(device)

    if surgery and second_stage:
        old_model = PredModel(
            3, 1, 20, batch_size=batch_size, hidden_dim=900, with_texts=False
        ).to(device)
        save_dir = os.path.join(surgery, "model.pt")
        state = torch.load(open(save_dir, "rb"), map_location=torch.device(device))
        old_model.load_state_dict(state["model"])
        model = surgery_fn(old_model, model)
        del old_model

    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    start_epoch = 0

    if load_path is not None:
        state = torch.load(
            open(os.path.join(load_path, "model.pt"), "rb"),
            map_location=torch.device(device),
        )
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        start_epoch = state["epoch"]
    loader, _ = get_loader(
        os.path.join(root, "data/strokes-py3.npy"), batch_size, with_texts=second_stage
    )
    num_epochs = 100
    save_dir = f"{root}runs/{batch_size}_{lr}_{time.time()}"
    os.makedirs(save_dir)

    is_scaling = True
    scaler = torch.cuda.amp.GradScaler(enabled=is_scaling)

    def _single_step(x, label_mask, input_mask, seqs=None):
        x = x.to(device)
        labels = x[:, :, 0]
        label_mask = label_mask.to(device)
        input_mask = input_mask.to(device)
        if second_stage:
            seqs = seqs.to(device)
        with torch.cuda.amp.autocast(enabled=is_scaling):
            # forward pass
            out = model(x, seqs)

            # retain grads for clipping
            out.retain_grad()

            # compute likelihood and sse metric
            prob_t, e_t, sse = model.infer(
                out, x[:, :, 1:], input_mask, label_mask
            )  # shape = (b,s)

            # calculate loss
            prob_loss = -prob_t.sum()

            e_t_loss = binary_cross_entropy_with_logits(
                input=e_t[input_mask], target=labels[label_mask], reduction="none"
            )
            e_t_loss = e_t_loss.sum()

            # divide seq loss by number of batches
            loss = (prob_loss + e_t_loss) / len(label_mask.sum(1) > 0)

        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optim)

        # clip w grad and output grad
        out.grad.clamp_(-100, 100)
        clip_grad_norm_(model.parameters(), 10)

        # plot_grad_flow(model.named_parameters())

        scaler.step(optim)
        scaler.update()

        return prob_loss.detach(), e_t_loss.detach(), sse

    for epoch in tqdm(range(start_epoch, num_epochs)):
        model.train()
        epoch_loss = 0
        e_loss = 0
        p_loss = 0
        epoch_sse = 0

        for batch_val in loader:
            model.reset()
            all_x, all_label_mask, all_input_mask, seqs = batch_val
            prob_loss, e_t_loss, sse = _single_step(
                all_x,
                all_label_mask,
                all_input_mask,
                seqs=seqs,
            )

            e_loss += e_t_loss / batch_size
            p_loss += prob_loss / batch_size
            epoch_sse += sse

        len_batches = len(loader)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--second",
        action="store_true",
        default=False,
        help="Set this flag to train a model conditioned on the sequence text",
    )
    parser.add_argument(
        "--surgery",
        type=str,
        default=None,
        help="Use a 1st stage model's weights to initialize 2nd stage model's weights."
        + " Specify the path of 1st stage model in this argument. "
        + "Use only for 2nd stage models. This is ignored for 1st stage models.",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Directory path to restore the model and optimizer state from.",
    )
    args = parser.parse_args()
    train(args.second, args.surgery, args.load_path)
