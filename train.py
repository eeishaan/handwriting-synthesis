from models.prediction import PredModel
from data import get_loader

from tqdm import tqdm
import torch

from constants import BATCH_FIRST

from torch.nn.functional import binary_cross_entropy_with_logits


def train():
    batch_size = 128
    model = PredModel(2, 3, 20, batch_size=batch_size)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loader = get_loader("data/strokes-py2.npy", batch_size)
    num_epochs = 100
    bce_loss = torch.nn.BCELoss(reduction="mean")
    for epoch in tqdm(range(num_epochs)):
        for x, labels, mask, total_elem in loader:
            out = model(x)
            model.reset()
            prob_t, e_t, ls = model.infer(out, x)  # shape = (b,s)

            # calculate loss
            loss = (
                -(prob_t * mask).sum(-1).mean()
                + binary_cross_entropy_with_logits(
                    input=e_t, target=labels, weight=mask, reduction="sum"
                )
                / total_elem
            )

            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"Epoch {epoch}: Loss {loss.detach()}")


if __name__ == "__main__":
    train()
