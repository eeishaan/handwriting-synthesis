import sys

sys.path.append("..")
import torch
import torch.nn as nn
from constants import BATCH_FIRST
from torch.distributions.multivariate_normal import MultivariateNormal
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

    def infer(self, lstm_out, x):
        lstm_out, ls = pad_packed_sequence(lstm_out, batch_first=BATCH_FIRST)
        x, ls = pad_packed_sequence(x, batch_first=BATCH_FIRST)
        lstm_out = lstm_out[:, :-1, :]  # discard last one
        e_t = lstm_out[..., 0]
        mp = lstm_out[..., 1:]
        ws, means, log_std, corr = torch.split(mp, self.split_sizes, dim=-1)
        b, s, _ = means.shape
        ws = nn.LogSoftmax(-1)(ws).view(b, s, self.num_mixtures)
        means = means.view(b, s, self.num_mixtures, 2)
        std = log_std.exp().view(b, s, self.num_mixtures, 2)
        corr = corr.tanh().view(b, s, self.num_mixtures, 1)
        std_1 = std[..., -2:-1] ** 2
        std_2 = std[..., -1:] ** 2
        covariance_mat = torch.cat(
            [std_1, std_1 * std_2 * corr, std_1 * std_2 * corr, std_2], axis=-1
        ).view(b, s, self.num_mixtures, 2, 2)
        dist = MultivariateNormal(means, covariance_matrix=covariance_mat)
        # pred = dist.sample()
        prob = dist.log_prob(x[:, 1:, :].unsqueeze(-2))
        prob = ws + prob
        prob = torch.logsumexp(prob, -1)
        return prob, e_t, ls
