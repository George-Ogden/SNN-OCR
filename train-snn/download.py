import os
import shutil

import kaggle

from config import data_root

kaggle.api.authenticate()
directory = os.path.join(os.path.dirname(__file__), data_root)

kaggle.api.dataset_download_files("preatcher/standard-ocr-dataset", path=directory, unzip=True)

shutil.rmtree(os.path.join(directory, "data2"))
for split in ("train", "test"):
    split = f"{split}ing_data"
    os.rename(os.path.join(directory, "data", split), os.path.join(directory, split))
os.rmdir(os.path.join(directory, "data"))
