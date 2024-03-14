from typing import List, Optional, Tuple

import einops
import numpy as np
import snntorch as snn
import torch as th
import torch.nn as nn

from .config import classes, num_characters
from .image import Image


# Define Network
class SNN(nn.Module):
    num_steps = 10
    beta = 0.95

    def __init__(self, input_size: Tuple[int, int], num_outputs: int):
        super().__init__()

        # Initialize layers
        self.input_size = input_size
        self.num_outputs = num_outputs
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(0.2),
        )

        with th.no_grad():
            hidden_size = self.conv(th.zeros(1, 1, *input_size)).shape[-1]
            assert 0 <= hidden_size < 1024

        self.fc = nn.Linear(hidden_size, num_outputs * self.num_steps)
        assert 0 <= num_outputs * self.num_steps < 1024
        self.lif = snn.Leaky(beta=self.beta)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem = self.lif.init_leaky()

        # Record the final layer
        spk_rec = []
        mem_rec = []

        hidden = self.fc(self.conv(x))
        hidden = einops.rearrange(hidden, "b (h n) -> h b n", h=self.num_steps)
        for i in range(self.num_steps):
            spk, mem = self.lif(hidden[i], mem)
            spk_rec.append(spk)
            mem_rec.append(mem)

        return th.stack(spk_rec, dim=0), th.stack(mem_rec, dim=0)

    @th.no_grad()
    def predict(self, images: List[Image], batch_size: int = 64) -> th.Tensor:
        if len(classes) != self.num_outputs:
            raise ValueError(
                "Number of classes does not match the number of outputs. Please, train the model first."
            )
        self.eval()
        results = []
        for i in range(0, len(images), batch_size):
            batch = th.tensor(
                np.array([image.image for image in images[i : i + batch_size]], dtype=np.float32),
                device=self.fc.weight.device,
            )
            # Add channel dimension.
            batch = batch.unsqueeze(1)
            spk, _ = self(batch)
            log_probs = th.log(spk.sum(dim=0) + 1e-3)
            results.append(log_probs.cpu())
        results = th.cat(results)

        logits = th.full((*results.shape[:-1], num_characters), -th.inf, dtype=th.float32)
        logits.scatter_(
            -1,
            th.tensor(classes).expand(results.shape[:-1] + (len(classes),)),
            results,
        )
        return logits


# LSTM
class LSTM(nn.Module):
    hidden_size = 1024
    num_layers = 3

    def __init__(self, vocab_size: int):
        super(LSTM, self).__init__()
        self.encoder = nn.Sequential(
            nn.Embedding(vocab_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)
        self.decoder = nn.Linear(self.hidden_size, vocab_size)
        self.h0 = nn.Parameter(th.zeros(self.num_layers, 1, self.hidden_size))
        self.c0 = nn.Parameter(th.zeros(self.num_layers, 1, self.hidden_size))

    def forward(
        self, x: th.Tensor, hidden_state: Optional[Tuple[th.Tensor, th.Tensor]] = None
    ) -> Tuple[th.Tensor, th.Tensor]:
        if hidden_state is None:
            hidden_state = self.hidden_state(x.shape[0])
        x = self.encoder(x)
        x, hidden_state = self.lstm(x, hidden_state)
        x = self.decoder(x)
        return x, hidden_state

    def hidden_state(self, n: Optional[int] = None) -> th.Tensor:
        if n is None:
            return (self.h0.squeeze(-2), self.c0.squeeze(-2))
        hidden_state = th.tile(self.h0, (1, n, 1)), th.tile(self.c0, (1, n, 1))
        return hidden_state
