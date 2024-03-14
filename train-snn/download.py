import os
import shutil
import tempfile

import kaggle

from config import data_root

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
