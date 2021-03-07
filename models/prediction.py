import torch
import torch.nn as nn
from ..constants import BATCH_FIRST
from torch.distributions.multivariate_normal import MultivariateNormal


class PredModel(nn.Module):
    def __init__(
        self,
        input_size,
        layers,
        num_mixtures,
    ):
        output_dim = 1 + 6 * num_mixtures
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=output_dim,
            num_layers=layers,
            batch_first=BATCH_FIRST,
        )
        self.num_mixtures = num_mixtures

    def forward(self, x):
        # x is of shape (batch, seq, x)
        y_hat = self.lstm(x)
        return y_hat

    def infer(self, lstm_out, x):
        lstm_out = lstm_out[:, -1:, :]  # discard last one
        # e_t = torch.sigmoid(lstm_out[..., 0])
        mp = lstm_out[..., 1:]
        ws, means, log_std, corr = torch.split(
            mp, [1, 2, 2, 1] * self.num_mixtures, dim=-1
        )
        b, s, _ = means.shape
        ws = nn.LogSoftmax(-1)(ws).view(b, s, self.num_mixtures)
        means = means.view(b, s, self.num_mixtures, 2)
        std = log_std.exp().view(b, s, self.num_mixtures, 2)
        corr = corr.tanh().view(b, s, self.num_mixtures, 1)
        std_1 = std[..., -2] ^ 2
        std_2 = std[..., -1] ^ 2
        covariance_mat = torch.cat(
            [std_1, std_1 * std_2 * corr, std_1 * std_2 * corr, std_2], axis=-1
        ).view(b, s, 2, 2)
        dist = MultivariateNormal(means, covariance_matrix=covariance_mat)
        # pred = dist.sample()
        prob = dist.log_prob(x[:, 1:, :])
        prob = ws + prob
        prob = torch.logsumexp(prob, -1)
        return prob
