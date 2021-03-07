import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, dataloader

from constants import BATCH_FIRST


class StrokeDataset(Dataset):
    def __init__(self, file):
        self.data = np.load(file, allow_pickle=True)
        # TODO: normalize the offsets?

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        self.data[index]


def collate_sequence(batch):
    lens = list(map(len, batch))

    # pad them
    padded = pad_sequence(batch, batch_first=BATCH_FIRST)

    # spearate out the labels
    labels = padded[:, :, 0]
    coordinates = padded[:, :, 1:]

    # pack them
    packed = pack_padded_sequence(coordinates, lens, batch_first=BATCH_FIRST)

    # dispatch
    return packed, labels


def get_loader(file, batch_size):
    dataset = StrokeDataset(file)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sequence,
    )
    return loader