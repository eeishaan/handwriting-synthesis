import torch
from torch.functional import Tensor
import torch.nn as nn


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