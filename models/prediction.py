from math import log
import sys

sys.path.append("..")
import torch
import torch.nn as nn
from constants import BATCH_FIRST
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
import numpy as np


@torch.jit.script
def weird_sig(x):
    x = 1 + x.exp()
    x = x.pow(-1)
    return x


class SkipLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):

        super(SkipLSTM, self).__init__()
        size = [input_size] + [hidden_size] * num_layers
        self.layers = nn.ModuleList(
            nn.LSTM(
                size[i],
                size[i + 1],
                1,
                batch_first=BATCH_FIRST,
            )
            for i in range(num_layers)
        )
        self.final_projs = nn.ModuleList(
            nn.Linear(hidden_size, output_size, bias=False) for i in range(num_layers)
        )
        self.next_layer_proj = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size, bias=False) for i in range(num_layers)
        )
        self.input_proj = nn.ModuleList(
            nn.Linear(input_size, hidden_size) for i in range(num_layers)
        )
        self.bias = torch.nn.Parameter(torch.randn(1))

    def forward(self, x, hs=None):
        inp = x
        running_skip = 0
        h_stack = []  # if hs is None else [hs[0]]
        c_stack = []  # if hs is None else [hs[1]]
        for i, layer in enumerate(self.layers):
            inp = self.input_proj[i](x) + self.next_layer_proj[i](inp)
            output, (_h, _c) = (
                layer(inp)
                if hs is None
                else layer(inp, (hs[0][i : i + 1], hs[1][i : i + 1]))
            )
            h_stack.append(_h)
            c_stack.append(_c)
            inp = output
            running_skip = running_skip + self.final_projs[i](output)
        output = running_skip + self.bias
        return output, (torch.cat(h_stack), torch.cat(c_stack))


class PredModel(nn.Module):
    def __init__(
        self,
        input_size,
        layers,
        num_mixtures,
        batch_size,
        hidden_dim,
    ):
        super(PredModel, self).__init__()
        self.hidden_dim = hidden_dim
        output_dim = 1 + 6 * num_mixtures
        self.output_dim = output_dim
        self.lstm = SkipLSTM(
            input_size=input_size,
            hidden_size=self.hidden_dim,
            output_size=output_dim,
            num_layers=layers,
        )
        self.num_mixtures = num_mixtures
        self.split_sizes = list(np.array([1, 2, 2, 1]) * num_mixtures)
        self.num_layers = layers
        self.h_n = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, requires_grad=True
        )
        self.c_n = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, requires_grad=True
        )
        self.batch_size = batch_size

    # def to(self, device):
    #     x = super(PredModel, self).to(device)
    #     self.h_n = self.h_n.to(device)
    #     self.c_n = self.c_n.to(device)
    #     return x

    # @torch.no_grad()
    # def reset(self):
    #     # return
    #     device = self.h_n.device
    #     del self.h_n, self.c_n
    #     self.h_n = torch.zeros(
    #         self.num_layers,
    #         self.batch_size,
    #         self.output_dim,
    #         requires_grad=True,
    #         device=device,
    #     )
    #     self.c_n = torch.zeros(
    #         self.num_layers,
    #         self.batch_size,
    #         self.hidden_dim,
    #         requires_grad=True,
    #         device=device,
    #     )

    def forward(self, x):
        # x is of shape (batch, seq, x)
        # y_hat, (self.h_n, self.c_n) = self.lstm(
        #     x, (self.h_n.detach(), self.c_n.detach())
        # )
        y_hat, _ = self.lstm(x)
        return y_hat

    def _process_output(self, lstm_out):
        e_t = weird_sig(lstm_out[..., 0])
        mp = lstm_out[..., 1:]
        ws, means, log_std, corr = torch.split(mp, self.split_sizes, dim=-1)
        b, s, _ = means.shape
        ws = nn.LogSoftmax(-1)(ws).view(b, s, self.num_mixtures)
        means = means.view(b, s, self.num_mixtures, 2)
        std = log_std.exp().view(b, s, self.num_mixtures, 2)
        corr = corr.tanh().view(b, s, self.num_mixtures, 1)
        std_1 = std[..., -2:-1] ** 2
        std_2 = std[..., -1:] ** 2
        covariance_mat = (std_1, std_2, corr)
        return ws, means, covariance_mat, e_t

    @torch.no_grad()
    def generate(self, device):
        inp = torch.randn(2, device=device).unsqueeze(0).unsqueeze(0)
        h_n, c_n = (
            torch.zeros(self.num_layers, 1, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, 1, self.hidden_dim, device=device),
        )
        out = []
        for i in range(700):
            y_hat, (h_n, c_n) = self.lstm(inp, (h_n, c_n))
            ws, means, covariance_mat, e_t = self._process_output(y_hat)

            ws = ws.squeeze().exp()
            j = ws.argmax()
            x_nt = means[..., j, :]
            inp = x_nt
            x_nt = x_nt.squeeze(0)
            e_t[e_t > 0.5] = 1
            e_t[e_t <= 0.5] = 0
            out.append(torch.cat([e_t, x_nt], axis=1))
        return out

    @staticmethod
    def log_prob(points, means, std_1, std_2, rho):
        eps = 1e-7
        points = points.unsqueeze(1)
        std_1 = std_1.squeeze(-1)
        std_2 = std_2.squeeze(-1)
        rho = rho.squeeze(-1)
        t1 = (points[..., 0] - means[..., 0]) / (std_1 + eps)
        t2 = (points[..., 1] - means[..., 1]) / (std_2 + eps)
        z = t1 ** 2 + t2 ** 2 - 2 * rho * t1 * t2
        prob = -z / (2 * (1 - rho ** 2) + eps)
        t = 2 * 3.1415927410125732 * std_1 * std_2 * torch.sqrt(1 - rho ** 2) + eps
        log_prob = prob + torch.log(torch.pow(t, -1))
        return log_prob

    def infer(self, lstm_out, x, mask):
        lstm_out = lstm_out[:, :-1, :]  # discard last one

        ws, means, covariance_mat, e_t = self._process_output(lstm_out)

        # apply mask
        seq_len = mask.sum(-1)
        mask = mask.reshape(-1)
        means = means.reshape(-1, *means.shape[-2:])[mask]
        std_1, std_2, rho = covariance_mat
        std_1 = std_1.reshape(-1, *std_1.shape[-2:])[mask]
        std_2 = std_2.reshape(-1, *std_2.shape[-2:])[mask]
        rho = rho.reshape(-1, *rho.shape[-2:])[mask]

        new_ws = ws.reshape(-1, ws.shape[-1])[mask]
        new_x = x[:, 1:, :].reshape(-1, x.shape[-1])[mask]
        prob = PredModel.log_prob(new_x, means, std_1, std_2, rho)
        prob = new_ws + prob
        prob = torch.logsumexp(prob, -1)
        seq_prob = torch.stack(
            [x.sum() for x in torch.split(prob, seq_len.tolist(), 0)]
        ).mean()

        # eval mse for xs
        sse = self.sse_x(new_x.detach(), new_ws.detach(), means.detach())
        return seq_prob, e_t, sse

    @torch.no_grad()
    def sse_x(self, target, w, means):
        w_x = w.argmax(-1).repeat(1, 2).view(-1, 1, 2)
        pred = torch.gather(means, 1, w_x).squeeze(1)
        # return nn.functional.mse_loss(target, pred)
        return nn.functional.mse_loss(
            target[:, 0], pred[:, 0]
        ) + nn.functional.mse_loss(
            target[:, 1], pred[:, 1]
        )  # compute error for each dim separately
