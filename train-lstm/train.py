import itertools
import os
from typing import List

import torch as th
import torch.utils.data as data_utils
import torchtext.transforms as T
from torchdata.datapipes.iter import FileLister, FileOpener
from torchvision.transforms import v2

from config import batch_size, data_root, sequence_length, validation_size

th.manual_seed(0)

num_classes = 128 + 1  # ASCII + 1 for padding


def tokenizer(string: str) -> List[int]:
    return [ord(c) for c in string if ord(c) < num_classes - 1]


transforms = v2.Compose(
    [
        v2.Lambda(tokenizer),
        T.Truncate(sequence_length),
        T.ToTensor(num_classes),
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

for batch in val_dataloader:
    print(batch)
    print(transforms(batch).shape)
    break

for batch in train_dataloader:
    print(batch)
    break
