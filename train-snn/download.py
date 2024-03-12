import os
import shutil
import tempfile

import kaggle
from PIL import ImageOps
from torchvision import datasets
from tqdm import tqdm

from config import data_root, num_samples

kaggle.api.authenticate()
directory = os.path.join(os.path.dirname(__file__), data_root)
if os.path.exists(directory):
    shutil.rmtree(directory)
os.makedirs(directory, exist_ok=True)

with tempfile.TemporaryDirectory() as temp_dir:
    kaggle.api.dataset_download_files("preatcher/standard-ocr-dataset", path=temp_dir, unzip=True)

    for split in ("train", "test"):
        split = f"{split}ing_data"
        target_directory = os.path.join(directory, f"kaggle_{split}")
        if os.path.exists(target_directory):
            shutil.rmtree(target_directory)
        os.rename(os.path.join(temp_dir, "data", split), target_directory)

with tempfile.TemporaryDirectory() as temp_dir:
    idx_2_class = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    counts = {c: 0 for c in idx_2_class}
    for train in (True, False):
        dir_name = ("train" if train else "test") + "ing_data"
        dataset = datasets.EMNIST(temp_dir, train=train, download=True, split="byclass")
        for i, (image, idx) in enumerate(tqdm(dataset)):
            # Limit number of EMNIST samples.
            if counts[idx_2_class[idx]] >= num_samples // 10:
                continue
            filename = os.path.join(
                directory, f"EMNIST_{dir_name}", idx_2_class[idx], f"{i:06}.png"
            )
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            ImageOps.invert(image).save(filename)
            counts[idx_2_class[idx]] += 1
