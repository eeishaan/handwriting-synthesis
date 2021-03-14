import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

from constants import BATCH_FIRST


class StrokeDataset(Dataset):
    def __init__(self, file, is_norm=True):
        global largest_sequence
        self.data = np.load(file, allow_pickle=True)

        data = list(self.data)
        data.sort(key=lambda x: x.shape[0])

        if is_norm:
            self.norm = StandardScaler()
            x = [s[:, 1:] for s in self.data]
            x_all = np.concatenate(x)
            self.norm.fit(x_all)
            for i, d in enumerate(data):
                t = self.norm.transform(d[:, 1:])
                t = np.concatenate([d[:, :1], t], axis=1)
                data[i] = d

        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        new_data = self.data[index]
        # new_data = np.pad(new_data, ((1, 0), (0, 0)))

        return torch.from_numpy(new_data)


def collate_sequence(batch):

    # data = list(self.data)
    batch.sort(key=lambda x: x.shape[0])

    lens = list(map(len, batch))

    # pad them
    padded = pad_sequence(batch, batch_first=BATCH_FIRST)

    # spearate out the labels
    labels = padded[:, :, 0]
    coordinates = padded[:, :, 1:]
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
    return coordinates, labels, label_mask, input_mask


def get_loader(file, batch_size):
    dataset = StrokeDataset(file)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sequence,
    )
    return loader, dataset