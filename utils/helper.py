import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from torch.functional import Tensor


def mult_normal_sample(mean, std_1, std_2, corr):
    std_1.squeeze_()
    std_2.squeeze_()
    corr.squeeze_()
    L = torch.cholesky(
        torch.as_tensor(
            [
                std_1 * std_2,
                std_1 * std_2 * corr,
                std_1 * std_2 * corr,
                std_1 * std_2,
            ]
        ).view(2, 2)
    )
    return mean + (torch.randn(1, 2) @ L).squeeze()


@torch.jit.script
def custom_logistic(x):
    ans = 1 / (1 + x.exp())
    return ans


@torch.jit.script
def phi_u_fn(u: Tensor, alpha: Tensor, beta: Tensor, keta: Tensor):
    ans = alpha - (beta * ((keta - u) ** 2))
    ans = ans.exp().sum(1)
    return ans


@torch.no_grad()
def sse_x(target, w, means):
    w_x = w.argmax(-1)
    pred = means[torch.arange(w_x.shape[0]), w_x, :]
    return nn.functional.mse_loss(target[:, 0], pred[:, 0]) + nn.functional.mse_loss(
        target[:, 1], pred[:, 1]
    )  # compute error for each dim separately


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
