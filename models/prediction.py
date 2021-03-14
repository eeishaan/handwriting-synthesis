from math import log
import sys
from numpy.lib.arraysetops import isin

sys.path.append("..")
import torch
import torch.nn as nn
from constants import BATCH_FIRST
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import numpy as np


@torch.jit.script
def weird_sig(x):
    x = 1 + x.exp()
    x = x.pow(-1)
    return x


class SkipLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):

        super(SkipLSTM, self).__init__()
        size = [hidden_size] + [hidden_size] * num_layers
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
        # self.next_layer_proj = nn.ModuleList(
        #     nn.Linear(hidden_size, hidden_size, bias=False) for i in range(num_layers)
        # )
        self.input_proj = nn.ModuleList(
            nn.Linear(input_size, hidden_size) for i in range(num_layers)
        )
        self.bias = torch.nn.Parameter(torch.randn(1))

    def forward(self, x, hs=None):
        last_inp = None
        running_skip = 0
        h_stack = []  # if hs is None else [hs[0]]
        c_stack = []  # if hs is None else [hs[1]]
        for i, layer in enumerate(self.layers):
            inp = self.input_proj[i](x)
            if last_inp is not None:
                inp = inp + last_inp
            output, (_h, _c) = (
                layer(inp)
                if hs is None
                else layer(inp, (hs[0][i : i + 1].detach(), hs[1][i : i + 1].detach()))
            )
            h_stack.append(_h)
            c_stack.append(_c)
            last_inp = output
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
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_dim,
            proj_size=self.output_dim,
            num_layers=layers,
            batch_first=BATCH_FIRST,
        )

        # SkipLSTM(
        #     input_size=input_size,
        #     hidden_size=self.hidden_dim,
        #     output_size=output_dim,
        #     num_layers=layers,
        # )
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

        self.reset()
        self.init_parameters("orthogonal")

    def init_parameters(self, init_type, gain=1):
        """Initialize model parameters
        Args:
            init_type (str, optional): type of initialization of weights; ["xavier","orthogonal"].
            gain (int, optional): optional scaling factor. Defaults to 1.
        """
        for param in self.parameters():
            if isinstance(param, nn.Linear):
                nn.init.xavier_uniform_(param.weight)
                param.bias.data.fill_(0.01)
            elif param.dim() == 1:
                nn.init.constant_(param, 0.0)
            elif param.dim() > 1:
                if init_type == "xavier":
                    nn.init.xavier_uniform_(param, gain=gain)
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(param, gain=gain)

    def reset(self):
        self.hidden = (
            torch.zeros(
                self.num_layers, self.batch_size, self.output_dim, requires_grad=True
            ),
            torch.zeros(
                self.num_layers, self.batch_size, self.hidden_dim, requires_grad=True
            ),
        )

    def forward(self, x):
        y_hat, _ = self.lstm(x)
        y_hat, _ = pad_packed_sequence(y_hat, batch_first=BATCH_FIRST)
        return y_hat

    def _process_output(self, lstm_out):
        e_t = weird_sig(lstm_out[..., 0])
        mp = lstm_out[..., 1:]
        ws, means, log_std, corr = torch.split(mp, self.split_sizes, dim=-1)
        b, s, _ = means.shape
        ws = nn.LogSoftmax(-1)(ws).view(b, s, self.num_mixtures)
        means = means.view(b, s, self.num_mixtures, 2)
        std = log_std.exp().view(b, s, self.num_mixtures, 2)
        corr = corr.tanh().view(b, s, self.num_mixtures)
        std_1 = std[..., -2]
        std_2 = std[..., -1]
        covariance_mat = (std_1, std_2, corr)
        return ws, means, covariance_mat, e_t

    @torch.no_grad()
    def generate(self, device):
        inp = torch.rand(2, device=device).unsqueeze(0).unsqueeze(0)
        h_n, c_n = (
            torch.zeros(self.num_layers, 1, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, 1, self.hidden_dim, device=device),
        )
        out = []
        for i in range(700):
            y_hat, (h_n, c_n) = self.lstm(inp, (h_n, c_n))
            ws, means, (std_1, std_2, corr), e_t = self._process_output(y_hat)

            ws = ws.squeeze().exp()
            j = ws.argmax()
            x_nt = means[..., j, :]
            inp = x_nt
            x_nt = x_nt.squeeze(0)
            u = torch.rand(1)[0]
            e_t[e_t <= u] = 1
            e_t[e_t > u] = 0

            out.append(torch.cat([e_t, x_nt], axis=1))
        return out

    @staticmethod
    def log_prob(points, means, std_1, std_2, rho):
        eps = 1e-7
        points = points.unsqueeze(1)
        t1 = (points[..., 0] - means[..., 0]) / (std_1 + eps)
        t2 = (points[..., 1] - means[..., 1]) / (std_2 + eps)
        z = t1 ** 2 + t2 ** 2 - 2 * rho * t1 * t2
        prob = -z / (2 * (1 - rho ** 2) + eps)
        t = 2 * 3.1415927410125732 * std_1 * std_2 * torch.sqrt(1 - rho ** 2) + eps
        log_prob = prob - torch.log(t)
        return log_prob

    def infer(self, lstm_out, x, input_mask, label_mask):
        ws, means, covariance_mat, e_t = self._process_output(lstm_out)
        x, _ = pad_packed_sequence(x, batch_first=BATCH_FIRST)

        # (batch, seq, mixs)

        # apply mask
        means = means[input_mask].reshape(-1, *means.shape[-2:])
        std_1, std_2, rho = covariance_mat
        std_1 = std_1[input_mask]
        std_2 = std_2[input_mask]
        rho = rho[input_mask]

        new_ws = ws[input_mask]
        new_x = x[label_mask]
        prob = PredModel.log_prob(new_x, means, std_1, std_2, rho)
        prob = new_ws + prob
        prob = torch.logsumexp(prob, -1)
        seq_prob = prob.sum() / len(label_mask.sum(1) > 0)

        # eval sse for xs
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
