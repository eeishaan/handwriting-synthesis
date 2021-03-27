import sys
from torch._C import dtype

from torch.functional import Tensor

sys.path.append("..")
import torch
import torch.nn as nn
from torch.nn import Module
from constants import BATCH_FIRST, EMBED_DIM
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
import numpy as np
from typing import Optional, Union, List, Tuple


@torch.jit.script
def weird_sig(x):
    # ans = torch.where(x >= 0, (-x).exp() / (1 + (-x).exp()), 1 / (1 + x.exp()))
    ans = 1 / (1 + x.exp())
    return ans


@torch.jit.script
def _phi_u(u: Tensor, alpha: Tensor, beta: Tensor, keta: Tensor):
    # u : (s_u)
    # alpha: (b, 10, 1)
    # u = u.unsqueeze(1)
    ans = alpha - (beta * ((keta - u) ** 2))
    ans = ans.exp().sum(1)
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
        self.output_size = output_size
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
            nn.Linear(hidden_size, output_size) for i in range(num_layers)
        )
        # self.next_layer_proj = nn.ModuleList(
        #     nn.Linear(hidden_size, hidden_size, bias=False) for i in range(num_layers)
        # )
        self.input_proj = nn.ModuleList(
            nn.Linear(input_size, hidden_size) for i in range(num_layers)
        )
        # self.input_proj = nn.Linear(input_size, hidden_size)
        self.bias = torch.nn.Parameter(torch.randn(1))

        if self.with_texts:
            self.c_size = 1
            self.window_proj = nn.Linear(hidden_size, win_size * self.c_size * 3)
            self.win_splits: List[int] = list(
                np.array([self.c_size, self.c_size, self.c_size]) * win_size
            )
            self.win_hidden_proj = nn.Linear(EMBED_DIM, hidden_size, bias=False)
            self.layers[0] = nn.LSTMCell(hidden_size, hidden_size)
        self.init_parameters("xavier")

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

    def compute_wt(self, seqs, h, _last_keta):
        # seqs: shape(b, s_u, 57)
        h = h.squeeze(1)
        # assert torch.isfinite(h).all()
        proj = self.window_proj(h)
        # assert torch.isfinite(proj).all()
        b, _ = proj.shape
        # b, s, mixtures
        win_splits = [10, 10, 10]
        alpha, beta, keta = torch.split(proj, win_splits, dim=-1)
        alpha = alpha.view(b, self.win_size, -1)
        beta = beta.exp().view(b, self.win_size, -1)
        keta = _last_keta + keta.exp().view(b, self.win_size, -1)

        u_len = seqs.shape[1]
        phi_u = _phi_u(
            torch.arange(start=1, end=u_len + 1, device=alpha.device),
            alpha,
            beta,
            keta,
        )
        # assert torch.isfinite(phi_u).all()
        # shape = b, s, s_u
        w_t = phi_u.unsqueeze(-1) * seqs
        # if isinstance(seq_mask, Tensor):
        #     w_t = w_t * seq_mask
        w_t = w_t.sum(-2)
        res = (w_t.unsqueeze(1), keta, phi_u)  # shape b, 57
        return res

    @staticmethod
    def _lstm_step(
        layer: torch.nn.modules.rnn.LSTMCell, hs: Optional[Tuple[Tensor, Tensor]], inp
    ):
        return layer.forward(inp) if hs is None else layer.forward(inp, (hs[0], hs[1]))

    def _one_step_wt(self, x, _hs: Optional[Tuple[Tensor, Tensor]], seqs, _last_keta):
        # assert torch.isfinite(x).all()
        _hs = self._lstm_step(self.layers[0], _hs, x)
        h_1 = _hs[0]
        res = self.compute_wt(seqs, h_1, _last_keta)
        w_1 = self.win_hidden_proj(res[0] if isinstance(res, tuple) else res)
        # assert torch.isfinite(w_1).all()
        ret = (w_1, h_1, _hs, res[1], res[2])
        return ret

    def run_first_layer(self, x, _hs: Optional[Tuple[Tensor, Tensor]], seqs):
        #  -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor], Tensor]:
        _, s, _ = x.shape
        last_w = 0  # torch.zeros_like(x[:, 0:1, :]).detach()

        h_final = torch.zeros_like(x)
        w_t = torch.zeros_like(x)
        _last_keta = 0  # torch.zeros(x.shape[0], 10, 1, dtype=x.dtype, device=x.device)
        for start in range(s):
            inp = x[:, start : start + 1, :] + last_w
            inp = inp.squeeze(1)
            # try:
            last_w, h_1, _hs, _last_keta, _ = self._one_step_wt(
                inp, _hs, seqs, _last_keta=_last_keta
            )
            # except Exception as e:
            #     print(x[:, start : start + 1, :])
            #     raise e
            h_final[:, start, :] = h_1
            w_t[:, start : start + 1, :] = last_w
        # if is_transfer:
        #     w_t = w_t.to(orig_device)
        #     h_final = h_final.to(orig_device)
        output_proj = self.final_projs[0](h_final)
        return w_t, h_final, _hs, output_proj

    def forward(
        self,
        x,
        hs: Optional[Tuple[Tensor, Tensor]] = None,
        seqs: Tensor = torch.zeros(1),
    ):
        last_inp: Optional[Tensor] = None
        running_skip = 0
        #  torch.zeros(
        #     x.shape[0],
        #     x.shape[1],
        #     self.output_size,
        #     device=x.device,
        #     dtype=x.dtype,
        # )
        h_stack = []  # if hs is None else [hs[0]]
        c_stack = []  # if hs is None else [hs[1]]
        # x = self.input_proj[0](x)
        w_t = 0
        for i, (layer, input_proj, final_proj) in enumerate(
            zip(self.layers, self.input_proj, self.final_projs)
        ):
            _hs = None
            if hs is not None:
                # print(type(hs[0][i]))
                _hs = (hs[0][i].detach(), hs[1][i].detach())

            inp = input_proj(x)
            if i == 0 and self.with_texts:
                # send only 1 time step
                # try:
                w_t, h_final, _, output_proj = self.run_first_layer(inp, None, seqs)
                # except Exception as e:
                #     print(x)
                #     print(inp)
                #     raise e
                last_inp = h_final
                running_skip = running_skip + output_proj
                # _h, _c = _hs
                # h_stack.append(_h)
                # c_stack.append(_c)
            else:
                if last_inp is not None:
                    inp = inp + last_inp
                    if self.with_texts:
                        inp = inp + w_t
                output, (_h, _c) = self._lstm_step(layer, _hs, inp)
                h_stack.append(_h)
                c_stack.append(_c)
                last_inp = output
                running_skip = running_skip + final_proj(output)
        output = running_skip + self.bias
        return output, (h_stack, c_stack)


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
        # self.lstm = torch.jit.script(self.lstm)
        # self.lstm2 = nn.LSTM(
        #     input_size=input_size,
        #     hidden_size=self.hidden_dim,
        #     proj_size=output_dim,
        #     num_layers=layers,
        #     batch_first=BATCH_FIRST,
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

        self.hidden = None

    def reset(self):
        self.hidden = None

    def forward(self, x, seqs=None, seq_mask=None):

        y_hat, self.hidden = self.lstm(x, None, seqs=seqs)
        # y_hat, self.hidden = self.lstm2(x, self.hidden)
        return y_hat

    def _process_output(self, lstm_out, bias=0):
        e_t = lstm_out[..., 0]
        # e_t = weird_sig(lstm_out[..., 0])
        mp = lstm_out[..., 1:]
        ws, means, log_std, corr = torch.split(mp, self.split_sizes, dim=-1)
        b, s, _ = means.shape
        ws = nn.LogSoftmax(-1)(ws * (1 + bias)).view(b, s, self.num_mixtures)
        means = means.view(b, s, self.num_mixtures, 2)
        std = (log_std - bias).exp().view(b, s, self.num_mixtures, 2)
        corr = corr.tanh().view(b, s, self.num_mixtures)
        std_1 = std[..., -2]
        std_2 = std[..., -1]
        covariance_mat = (std_1, std_2, corr)
        return ws, means, covariance_mat, e_t

    @torch.no_grad()
    def generate_with_seq(self, seed, device, seqs, bias=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        x = torch.zeros(3, device=device).unsqueeze(0).unsqueeze(0)
        # _hs = (
        #     torch.zeros(self.num_layers, 1, self.hidden_dim, device=device),
        #     torch.zeros(self.num_layers, 1, self.hidden_dim, device=device),
        # )
        out = [x]
        last_w = 0
        hs = None
        # stop_vec = torch.zeros(1, 1, EMBED_DIM, device=device)
        # stop_vec[..., -1] = 1
        # seqs = torch.cat([seqs, stop_vec], axis=1)
        idx = 0
        _last_keta = 0
        while True:
            h_stack = []
            c_stack = []
            inp = self.lstm.input_proj[0](x) + last_w
            inp.squeeze_(1)
            _hs = (hs[0][0].detach(), hs[1][0].detach()) if hs else None
            # last_w, h_1, _hs, phi
            last_w, h_1, _hs, _last_keta, phi = self.lstm._one_step_wt(
                inp, _hs, seqs, _last_keta=_last_keta
            )
            h_stack.append(_hs[0])
            c_stack.append(_hs[1])
            # if idx > 700:  # (phi[0, -1] > phi[0, :-1]).all():
            if (phi[0, -1] > phi[0, :-1]).all():
                break
            idx += 1
            h_1 = h_1.unsqueeze(1)
            running_skip = self.lstm.final_projs[0](h_1)

            last_inp = h_1
            inp = self.lstm.input_proj[0](x) + last_w
            for i, layer in enumerate(self.lstm.layers[1:]):
                _hs = (hs[0][i + 1].detach(), hs[1][i + 1].detach()) if hs else None
                layer_inp = inp + last_inp
                output, (_h, _c) = self.lstm._lstm_step(layer, _hs, layer_inp)
                h_stack.append(_h)
                c_stack.append(_c)
                last_inp = output
                running_skip = running_skip + self.lstm.final_projs[i + 1](output)
            y_hat = running_skip + self.lstm.bias
            hs = (h_stack, c_stack)

            ws, means, (std_1, std_2, corr), e_tt = self._process_output(
                y_hat, bias=bias
            )

            ws = ws.squeeze().exp()
            j = torch.multinomial(ws, 1)[0]
            # j = ws.argmax()
            x_nt = means[..., j, :]
            # u = torch.rand(1)[0]
            u = 0.5
            # e_t = e_tt
            e_tt = e_tt.sigmoid()
            e_t = torch.zeros_like(e_tt)
            e_t[e_tt <= u] = 0
            e_t[e_tt > u] = 1
            x = torch.cat([e_t.unsqueeze(0), x_nt], axis=-1)
            out.append(x)
        return out

    @torch.no_grad()
    def generate(self, seed, device):
        torch.manual_seed(seed)
        np.random.seed(seed)
        inp = torch.zeros(3, device=device).unsqueeze(0).unsqueeze(0)
        # h_n, c_n = (
        #     torch.zeros(self.num_layers, 1, self.hidden_dim, device=device),
        #     torch.zeros(self.num_layers, 1, self.hidden_dim, device=device),
        # )
        hs = None
        out = []
        for i in range(700):
            y_hat, hs = self.lstm(inp, hs)
            # h_n = torch.cat(h_n)
            # c_n = torch.cat(c_n)
            ws, means, (std_1, std_2, corr), e_tt = self._process_output(y_hat)

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
            u = 0.5  # torch.rand(1)[0]
            # e_t = e_t.log()
            # print(e_t)
            # e_tt = e_tt.sigmoid()
            e_tt = weird_sig(e_tt)
            e_t = torch.zeros_like(e_tt)
            e_t[e_tt <= u] = 0
            e_t[e_tt > u] = 1
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
        t = (
            np.log(2 * 3.1415927410125732)
            + std_1.log()
            + std_2.log()
            + 0.5 * torch.log(1 - rho ** 2)
        )
        log_prob = prob - t
        return log_prob

    def infer(self, lstm_out, x, input_mask, label_mask):
        ws, means, covariance_mat, e_t = self._process_output(lstm_out)

        # (batch, seq, mixs)

        # apply mask
        means = means[input_mask]  # .reshape(-1, *means.shape[-2:])
        std_1, std_2, rho = covariance_mat
        std_1 = std_1[input_mask]
        std_2 = std_2[input_mask]
        rho = rho[input_mask]

        new_ws = ws[input_mask]
        new_x = x[label_mask]
        prob = PredModel.log_prob(new_x, means, std_1, std_2, rho)
        prob = new_ws + prob
        prob = torch.logsumexp(prob, -1)
        seq_prob = prob

        # eval sse for xs
        sse = self.sse_x(new_x.detach(), new_ws.detach(), means.detach())
        return seq_prob, e_t, sse

    @torch.no_grad()
    def sse_x(self, target, w, means):
        w_x = w.argmax(-1)
        pred = means[torch.arange(w_x.shape[0]), w_x, :]
        # pred = torch.gather(means, 1, w_x).squeeze(1)
        # return nn.functional.mse_loss(target, pred)
        return nn.functional.mse_loss(
            target[:, 0], pred[:, 0]
        ) + nn.functional.mse_loss(
            target[:, 1], pred[:, 1]
        )  # compute error for each dim separately
