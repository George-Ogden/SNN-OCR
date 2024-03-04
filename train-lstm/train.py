import itertools
import os

import torch as th
import torch.utils.data as data_utils
from torchdata.datapipes.iter import FileLister, FileOpener

from config import batch_size, data_root, validation_size

th.manual_seed(0)

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
    break

for batch in train_dataloader:
    print(batch)
    break
