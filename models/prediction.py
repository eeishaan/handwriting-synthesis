import sys

sys.path.append("..")
import torch
import torch.nn as nn
from constants import BATCH_FIRST
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np


class PredModel(nn.Module):
    def __init__(
        self,
        input_size,
        layers,
        num_mixtures,
        batch_size,
    ):
        super(PredModel, self).__init__()
        output_dim = 1 + 6 * num_mixtures
        self.output_dim = output_dim
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=output_dim,
            num_layers=layers,
            batch_first=BATCH_FIRST,
        )
        self.num_mixtures = num_mixtures
        self.split_sizes = list(np.array([1, 2, 2, 1]) * num_mixtures)
        self.num_layers = layers
        self.h_n = torch.zeros(
            self.num_layers, batch_size, self.output_dim, requires_grad=True
        )
        self.c_n = torch.zeros(
            self.num_layers, batch_size, self.output_dim, requires_grad=True
        )

    def reset(self):
        self.h_n = torch.zeros_like(self.h_n)
        self.c_n = torch.zeros_like(self.c_n)

    def forward(self, x):
        # x is of shape (batch, seq, x)
        y_hat, (self.h_n, self.c_n) = self.lstm(x)
        return y_hat

    def _process_output(self, lstm_out):
        e_t = lstm_out[..., 0]
        mp = lstm_out[..., 1:]
        ws, means, log_std, corr = torch.split(mp, self.split_sizes, dim=-1)
        b, s, _ = means.shape
        ws = nn.LogSoftmax(-1)(ws).view(b, s, self.num_mixtures)
        means = means.view(b, s, self.num_mixtures, 2)
        std = log_std.exp().view(b, s, self.num_mixtures, 2)
        corr = corr.tanh().view(b, s, self.num_mixtures, 1)  # + 1e-7
        std_1 = std[..., -2:-1] ** 2  # + 1e-7
        std_2 = std[..., -1:] ** 2  # + 1e-7
        covariance_mat = torch.cat(
            [std_1, std_1 * std_2 * corr, std_1 * std_2 * corr, std_2], axis=-1
        ).view(b, s, self.num_mixtures, 2, 2)

        return ws, means, covariance_mat, e_t

    @torch.no_grad()
    def generate(self, device):
        inp = torch.zeros(2, device=device).unsqueeze(0).unsqueeze(0)
        h_n, c_n = torch.zeros(self.num_layers, 1, self.output_dim), torch.zeros(
            self.num_layers, 1, self.output_dim
        )
        out = []
        for _ in range(700):
            y_hat, (h_n, c_n) = self.lstm(inp, (h_n, c_n))

            ws, means, covariance_mat, e_t = self._process_output(y_hat)

            ws = ws.squeeze().exp()

            dist = Categorical(probs=ws)
            j = dist.sample()
            dist = MultivariateNormal(
                means[..., j, :], covariance_matrix=covariance_mat[..., j, :, :]
            )
            x_nt = dist.sample()
            x_nt = x_nt.squeeze(0)
            e_t = e_t.sigmoid()
            e_t[e_t > 0.5] = 1
            e_t[e_t <= 0.5] = 0
            out.append(torch.cat([e_t, x_nt], axis=1))
        return out

    def infer(self, lstm_out, x, mask):
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=BATCH_FIRST)
        x, _ = pad_packed_sequence(x, batch_first=BATCH_FIRST)
        lstm_out = lstm_out[:, :-1, :]  # discard last one

        ws, means, covariance_mat, e_t = self._process_output(lstm_out)

        # apply mask
        # new_mask = mask[:, :-1]
        seq_len = mask.sum(-1)
        mask = mask.reshape(-1)
        means = means.reshape(-1, *means.shape[-2:])[mask]
        covariance_mat = covariance_mat.reshape(-1, *covariance_mat.shape[-3:])[mask]

        dist = MultivariateNormal(means, covariance_matrix=covariance_mat)
        # pred = dist.sample()
        new_ws = ws.reshape(-1, ws.shape[-1])[mask]
        new_x = x[:, 1:, :].reshape(-1, x.shape[-1])[mask]
        prob = dist.log_prob(new_x.unsqueeze(-2))
        prob = new_ws + prob
        prob = torch.logsumexp(prob, -1)
        seq_prob = torch.stack(
            [x.sum() for x in torch.split(prob, seq_len.tolist(), 0)]
        ).mean()
        return seq_prob, e_t
