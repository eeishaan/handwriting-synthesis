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
    ans = torch.where(x >= 0, (-x).exp() / (1 + (-x).exp()), 1 / (1 + x.exp()))
    return ans


class SkipLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        with_texts=False,
        win_size=None,
    ):

        super(SkipLSTM, self).__init__()
        self.with_texts = with_texts
        self.win_size = win_size
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
        # self.input_proj = nn.ModuleList(
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.bias = torch.nn.Parameter(torch.randn(1))

        if self.with_texts:
            self.c_size = 1
            self.window_proj = nn.Linear(hidden_size, win_size * self.c_size * 3)
            self.win_splits = list(
                np.array([self.c_size, self.c_size, self.c_size]) * win_size
            )
            self.win_hidden_proj = nn.Linear(57, hidden_size, bias=False)

    def compute_wt(self, seqs, h):
        # seqs: shape(b, s_u, 57)
        def _phi_u(u, alpha, beta, keta):
            # u : (s_u)
            # alpha: (b, s, 10, 1)
            # u = u.unsqueeze(1)
            ans = alpha - beta * (keta - u) ** 2
            ans = ans.sum(-2)
            return ans

        proj = self.window_proj(h)
        b, s, _ = proj.shape
        # b, s, mixtures
        alpha, beta, keta = torch.split(proj, self.win_splits, dim=-1)
        alpha = alpha.view(b, s, self.win_size, -1)
        beta = beta.exp().view(b, s, self.win_size, -1)
        keta = keta.exp().view(b, s, self.win_size, -1).cumsum(1)

        u_len = seqs.shape[1]
        phi_u = _phi_u(torch.arange(u_len, device=alpha.device), alpha, beta, keta)
        # shape = b, s, s_u
        w_t = phi_u.unsqueeze(-1) * seqs.unsqueeze(1)
        w_t = w_t.sum(2)
        return w_t  # shape b, s, 57

    def forward(self, x, hs=None, seqs=None):
        last_inp = None
        running_skip = 0
        h_stack = []  # if hs is None else [hs[0]]
        c_stack = []  # if hs is None else [hs[1]]
        x = self.input_proj(x)

        def _lstm_step(layer, hs, inp):
            return layer(inp) if hs is None else layer(inp, (hs[0], hs[1]))

        for i, layer in enumerate(self.layers):
            _hs = (hs[0][i : i + 1].detach(), hs[1][i : i + 1].detach()) if hs else None
            if i == 0 and self.with_texts:
                # send only 1 time step
                _, s, _ = x.shape
                last_w = 0
                h_final = torch.zeros_like(x)
                w_t = torch.zeros_like(x)
                for start in range(s):
                    inp = last_w + x[:, start : start + 1, :]
                    h_1, _hs = _lstm_step(layer, _hs, inp)
                    w_1 = self.compute_wt(seqs, h_1)
                    last_w = self.win_hidden_proj(w_1)
                    h_final[:, start : start + 1, :] = h_1
                    w_t[:, start : start + 1, :] = last_w
                last_inp = h_final
                running_skip = running_skip + self.final_projs[i](last_inp)
                h_stack.append(_hs[0])
                c_stack.append(_hs[1])
                continue
            inp = x
            if last_inp is not None:
                inp = inp + last_inp
                if self.with_texts:
                    inp = inp + w_t
            output, (_h, _c) = _lstm_step(layer, _hs, inp)
            h_stack.append(_h)
            c_stack.append(_c)
            last_inp = output
            running_skip = running_skip + self.final_projs[i](output)
        output = running_skip + self.bias
        return output, (torch.cat(h_stack), torch.cat(c_stack))


class PredModel(nn.Module):
    def __init__(
        self, input_size, layers, num_mixtures, batch_size, hidden_dim, with_texts=False
    ):
        super(PredModel, self).__init__()
        self.hidden_dim = hidden_dim
        output_dim = 1 + 6 * num_mixtures
        self.output_dim = output_dim
        self.win_size = 10
        self.lstm = SkipLSTM(
            input_size=input_size,
            hidden_size=self.hidden_dim,
            output_size=output_dim,
            num_layers=layers,
            with_texts=with_texts,
            win_size=self.win_size,
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

        self.hidden = None

    def reset(self):
        self.hidden = None

    def forward(self, x, seqs=None):
        y_hat, self.hidden = self.lstm(x, self.hidden, seqs=seqs)
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
    def generate(self, seed, device, seqs=None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        inp = torch.zeros(3, device=device).unsqueeze(0).unsqueeze(0)
        h_n, c_n = (
            torch.zeros(self.num_layers, 1, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, 1, self.hidden_dim, device=device),
        )
        out = []
        for i in range(700):
            y_hat, (h_n, c_n) = self.lstm(inp, (h_n, c_n), seqs=seqs)
            ws, means, (std_1, std_2, corr), e_t = self._process_output(y_hat)

            ws = ws.squeeze().exp()
            # print(ws)
            j = torch.multinomial(ws, 1)[0]
            # j = ws.argmax()
            x_nt = means[..., j, :]
            # std_1 = std_1[..., j] ** 2
            # std_2 = std_2[..., j] ** 2
            # corr = corr[..., j] ** 2
            # covariance_mat = torch.cat(
            #     [std_1, std_1 * std_2 * corr, std_1 * std_2 * corr, std_2], axis=-1
            # ).view(1, 2, 2)
            # dist = MultivariateNormal(x_nt.squeeze(0), covariance_matrix=covariance_mat)
            # x_nt = dist.sample().unsqueeze(0)
            u = torch.rand(1)[0]
            e_t[e_t <= u] = 0
            e_t[e_t > u] = 1
            inp = torch.cat([e_t.unsqueeze(0), x_nt], axis=-1)

            out.append(inp)
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
