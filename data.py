import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, dataloader
from sklearn.preprocessing import Normalizer

from constants import BATCH_FIRST


class StrokeDataset(Dataset):
    def __init__(self, file):
        self.data = np.load(file, allow_pickle=True, encoding="latin1")
        self.norm = Normalizer()
        x = [s[:, 1:] for s in self.data]
        x = np.concatenate(x)
        self.norm.fit_transform(x)
        # TODO: normalize the offsets?

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        new_data = np.pad(self.data[index], ((0, 1), (0, 0)))
        new_data[:, -1] = 1
        # new_data = self.data[index]
        data = self.norm.transform(new_data[:, 1:])
        new_data = np.concatenate([new_data[:, 0:1], data], axis=1)

        return torch.from_numpy(new_data)


def collate_sequence(batch):
    lens = list(map(len, batch))

    # pad them
    padded = pad_sequence(batch, batch_first=BATCH_FIRST)

    # spearate out the labels
    labels = padded[:, :-1, 0]  # remove the phi's label
    coordinates = padded[:, :, 1:]

    # pack them
    packed = pack_padded_sequence(
        coordinates, lens, enforce_sorted=False, batch_first=BATCH_FIRST
    )
    # labels = pack_padded_sequence(
    #     labels, lens, enforce_sorted=False, batch_first=BATCH_FIRST
    # )

    max_len = padded.shape[1] - 1  # remove the extra phi at the end
    label_mask = torch.arange(max_len).expand(len(lens), max_len) < (
        torch.Tensor(lens) - 1
    ).unsqueeze(1)
    # label_mask = label_mask[:, :-1]
    # dispatch
    return packed, labels, label_mask


def get_loader(file, batch_size):
    dataset = StrokeDataset(file)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sequence,
    )
    return loader