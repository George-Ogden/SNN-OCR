import itertools
import os
import sys
from typing import List

import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torchtext.transforms as T
from torchdata.datapipes.iter import FileLister, FileOpener
from torchvision.transforms import v2
from tqdm import tqdm

from config import (
    batch_size,
    data_root,
    evaluation_interval,
    learning_rate,
    sequence_length,
    validation_size,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import save_directory
from src.model import LSTM

th.manual_seed(0)

vocab_size = 128 + 1  # ASCII + 1 for padding
pad_token = vocab_size - 1


def tokenizer(string: str) -> List[int]:
    return [0] + [ord(c) for c in string if ord(c) < vocab_size - 1]


transforms = v2.Compose(
    [
        v2.Lambda(tokenizer),
        T.Truncate(sequence_length + 1),
        T.ToTensor(pad_token),
    ]
)

data_path = os.path.join(os.path.dirname(__file__), data_root)
dp = FileLister(root=data_path).filter(lambda fname: fname.endswith(".txt"))
dp = FileOpener(dp)

dp = dp.readlines(strip_newline=True, return_path=False).shuffle().sharding_filter()

val_dataset = list(itertools.islice(dp, validation_size))

val_dataloader = data_utils.DataLoader(
    val_dataset,
    batch_size=batch_size,
)
train_dataloader = data_utils.DataLoader(
    dp,
    batch_size=batch_size,
)

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# Instantiate the network
model = LSTM(vocab_size).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_loss = float("inf")
model_path = os.path.join(save_directory, "lstm.pth")

t1 = tqdm(
    enumerate(train_dataloader), desc="Training", postfix={"Val loss": best_val_loss, "Loss": 0}
)
for i, batch in t1:
    if i % evaluation_interval == 0:
        with th.no_grad():
            # Evaluate
            model.eval()
            val_loss = 0
            t2 = tqdm(val_dataloader, desc="Evaluating", leave=False)
            for val_batch in t2:
                sequence = transforms(val_batch).to(device)
                sequence[:, 0] = pad_token
                inputs = sequence[:, :-1]
                targets = sequence[:, 1:]

                outputs, _ = model(inputs)
                val_loss += loss_fn(outputs.permute(0, 2, 1), targets).item()
            val_loss /= len(val_dataloader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                th.save(model.state_dict(), model_path)
            model.train()

    sequence = transforms(batch).to(device)
    sequence[:, 0] = pad_token
    inputs = sequence[:, :-1]
    targets = sequence[:, 1:]

    outputs, _ = model(inputs)
    loss = loss_fn(outputs.permute(0, 2, 1), targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    t1.set_postfix({"Loss": loss.item(), "Val loss": best_val_loss})
