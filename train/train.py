import cv2
import numpy as np
import torch as th
import torch.utils.data as data_utils
import torchvision.datasets as datasets
from config import batch_size, data_root, image_size
from torchvision.transforms import v2

th.manual_seed(0)
np.random.seed(0)

# Define a transform
transform = v2.Compose(
    [v2.Resize(image_size), v2.Grayscale(), v2.ToTensor(), v2.Normalize((0,), (1,))]
)
train_transform = v2.Compose(
    [
        transform,
        v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
    ]
)

# Load the dataset
train_dataset = datasets.ImageFolder(data_root, transform=train_transform)
val_dataset = datasets.ImageFolder(data_root, transform=transform)

# Create data subsets.
indices = th.randperm(len(train_dataset)).tolist()
train_indices = indices[: int(len(indices) * 0.9)]
val_indices = indices[int(len(indices) * 0.9) :]

# Create DataLoaders
train_loader = data_utils.DataLoader(
    data_utils.Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True
)
val_loader = data_utils.DataLoader(
    data_utils.Subset(val_dataset, val_indices), batch_size=batch_size
)

for x in train_loader:
    for i in range(batch_size):
        img = x[0][i].numpy().reshape(*image_size)
        label = x[1][i]
        cv2.imshow("image", img)
        print(chr(int(train_dataset.classes[label])))
        cv2.waitKey(1000)
    break
