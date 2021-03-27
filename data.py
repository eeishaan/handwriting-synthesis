import os
import string

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from constants import BATCH_FIRST, EMBED_DIM

L_MAP = {c: label for label, c in enumerate(string.ascii_letters)}


def get_embedding(l):
    def get_label(x):
        return L_MAP.get(x, 56)

    labels = list(map(get_label, l))
    hot = np.zeros((len(labels), EMBED_DIM), dtype=np.float32)
    hot[np.arange(len(labels)), labels] = 1
    stop_marker = np.zeros(EMBED_DIM)
    stop_marker[-1] = 1
    if l[-1] == "\n":
        hot[-1] = stop_marker
    else:
        hot = np.concatenate([hot, stop_marker[np.newaxis, :]], 0)
    return hot


class StrokeDataset(Dataset):
    def __init__(self, file, is_norm=True, with_texts=False):
        global largest_sequence
        self.data = np.load(file, allow_pickle=True)

        idxs = list(range(self.data.shape[0]))
        idxs.sort(key=lambda x: self.data[x].shape[0], reverse=True)
        data = self.data[idxs]

        self.with_texts = with_texts
        if is_norm:
            self.norm = StandardScaler()
            x = [s[:, 1:] for s in self.data]
            x_all = np.concatenate(x)
            self.norm.fit(x_all)
            for i, d in enumerate(data):
                t = self.norm.transform(d[:, 1:])
                t = np.concatenate([d[:, :1], t], axis=1)
                data[i] = t
        if with_texts:
            s_file = os.path.join(os.path.dirname(file), "sentences.txt")
            with open(s_file) as fob:
                texts = fob.readlines()

            texts = [texts[i] for i in idxs]
            self._lines = texts
            l_map = {c: label for label, c in enumerate(string.ascii_letters)}

            def get_label(x):
                return l_map.get(x, 56)

            one_hot = []
            for i, l in enumerate(texts):
                hot = get_embedding(l)
                one_hot.append(hot)
            self.texts = one_hot

        self.data = data
        self.chunk_len = None  # 700 if not self.with_texts else None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        new_data = self.data[index]

        if self.chunk_len:
            max_len = max(new_data.shape[0] - self.chunk_len, 0)
            start = 0
            if max_len > 0:
                start = np.random.randint(max_len)
            end = start + self.chunk_len
            new_data = new_data[start:end, :]
        new_data = np.pad(new_data, ((1, 0), (0, 0)))
        res = torch.from_numpy(new_data)
        if self.with_texts:
            res = (res, torch.from_numpy(self.texts[index]))
        return res


def collate_sequence(inp):

    # data = list(self.data)
    with_chars = isinstance(inp[0], tuple)

    if with_chars:

        def _cmp(x):
            return x[0].shape[0]

        inp.sort(key=_cmp, reverse=True)
        batch = [b[0] for b in inp]
        chars = [b[1] for b in inp]
        chars = pad_sequence(chars, batch_first=BATCH_FIRST)
    else:

        def _cmp(x):
            return x.shape[0]

        inp.sort(key=_cmp, reverse=True)
        batch = inp

    lens = list(map(len, batch))

    # pad them
    padded = pad_sequence(batch, batch_first=BATCH_FIRST)

    # spearate out the labels
    labels = padded[:, :, 0]
    coordinates = padded[:, :, :]
    # make label mask

    # remove one for training, as we can't predict for the last label
    max_len = padded.shape[1]

    label_mask = torch.arange(max_len).expand(len(lens), max_len) < (
        torch.Tensor(lens)
    ).unsqueeze(1)
    label_mask[:, 0] = False

    # max_len = -1
    input_mask = torch.arange(max_len).expand(len(lens), max_len) < (
        torch.Tensor(lens) - 1
    ).unsqueeze(1)

    # dispatch
    res = (
        coordinates,
        labels,
        label_mask,
        input_mask,
    )
    if with_chars:
        res += (chars,)
    return res


def get_loader(file, batch_size, with_texts=False):
    dataset = StrokeDataset(file, with_texts=with_texts)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sequence,
    )
    return loader, dataset
