import os
import sys

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision.datasets as datasets
from torchvision.transforms import v2
from tqdm import tqdm, trange

from config import batch_size, data_root, learning_rate, num_epochs

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import image_size, save_directory
from src.model import SNN

# Set random seeds
th.manual_seed(0)
np.random.seed(0)

# Define a transform
transform = v2.Compose(
    [
        v2.Resize(image_size),
        v2.Grayscale(),
        v2.ToImage(),
        v2.ToDtype(th.float32, scale=True),
        v2.Normalize((0,), (1,)),
    ]
)
train_transform = v2.Compose(
    [
        transform,
        v2.RandomAdjustSharpness(0.5),
        v2.RandomApply(
            [
                v2.GaussianBlur(5),
            ]
        ),
        v2.RandomChoice(
            [
                v2.RandomPerspective(),
                v2.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                v2.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            ]
        ),
        v2.RandomErasing(),
    ]
)

# Load the dataset
data_path = os.path.join(os.path.dirname(__file__), data_root)
train_dataset = datasets.ImageFolder(data_path, transform=train_transform)
val_dataset = datasets.ImageFolder(data_path, transform=transform)

# Create data subsets
indices = th.randperm(len(train_dataset)).tolist()
train_indices = indices[: int(len(indices) * 0.9)]
val_indices = indices[int(len(indices) * 0.9) :]

# Create DataLoaders
train_loader = data_utils.DataLoader(
    data_utils.Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True
)
val_loader = data_utils.DataLoader(data_utils.Subset(val_dataset, val_indices), batch_size=1024)

# Save dataset information
os.makedirs(save_directory, exist_ok=True)
with open(os.path.join(save_directory, "classes.txt"), "w") as f:
    f.write("\n".join(train_dataset.classes))

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# Instantiate the network
model = SNN(input_size=image_size, num_outputs=len(train_dataset.classes)).to(device)
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

best_loss = float("inf")
model_path = os.path.join(save_directory, "snn.pth")

t1 = trange(num_epochs, desc="Training", postfix={"Best Val Loss": best_loss, "Loss": 0})
for epoch in t1:
    # Minibatch training loop
    model.train()
    t2 = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for i, (data, targets) in enumerate(t2):
        data = data.to(device)
        targets = targets.to(device)

        spk_rec, mem_rec = model(data)

        # Initialize the loss & sum over time
        loss = th.zeros((), device=device)
        for step in range(model.num_steps):
            loss += loss_fn(mem_rec[step], targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t2.set_postfix({"Loss": loss.item()})

    count = 0
    total_loss = 0
    model.eval()
    t2 = tqdm(val_loader, desc="Evaluating", leave=False)
    with th.no_grad():
        for data, targets in t2:
            data = data.to(device)
            targets = targets.to(device)

            val_spk, _ = model(data)

            # Calculate Loss
            predicted = val_spk.sum(dim=0)
            loss = loss_fn(th.log(predicted + 1e-3), targets).item()
            total_loss += loss * targets.size(0)
            count += targets.size(0)
            t2.set_postfix({"Loss": loss})

        val_loss = total_loss / count

        if val_loss < best_loss:
            best_loss = val_loss
            th.save(model.state_dict(), model_path)
        t1.set_postfix({"Best Loss": best_loss, "Val Loss": val_loss})

print(f"Best Loss: {best_loss:.3f}")
