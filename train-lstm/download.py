import json
import os
import subprocess
import sys
from typing import Iterable

from tqdm import trange

from config import data_root

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

data_path = os.path.join(os.path.dirname(__file__), data_root)


def download_and_extract_all(url: str, num_shards: int, limit: int = float("inf")) -> Iterable[str]:
    for i in trange(min(num_shards, limit)):
        formatted_url = url.format(index=i, n_shards=num_shards)
        download_and_extract(formatted_url)


def download_and_extract(url: str) -> str:
    filename = download(url)
    if filename:
        filename = unzip(filename)
        filename = reformat(filename)
    return filename


def download(url: str) -> str:
    filename = os.path.basename(url)
    os.makedirs(data_path, exist_ok=True)
    path = os.path.join(data_path, filename)
    if not os.path.exists(path):
        subprocess.run(
            f"curl -L {url} -o {path}", shell=True, check=True, capture_output=True, text=True
        )
        return filename


def extract(filename):
    filename = unzip(filename)
    filename = reformat(filename)
    return filename


def unzip(filename: str) -> str:
    path = os.path.join(data_path, filename)
    subprocess.run(f"gzip -d {path}", shell=True, check=True, capture_output=True, text=True)
    # Create a file to indicate that the extraction is complete.
    with open(path, "x"):
        ...
    path, ext = os.path.splitext(path)
    return path


def reformat(filename: str) -> str:
    path, ext = os.path.splitext(filename)
    text_filename = path + ".txt"
    with open(filename, "r") as file, open(text_filename, "w") as text_file:
        for line in file:
            data = json.loads(line)
            text = data["text"]
            text_file.write(text + "\n")
    os.unlink(filename)
    return text_filename


if __name__ == "__main__":
    filenames = download_and_extract_all(
        "https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/en/c4-train.{index:05d}-of-{n_shards:05d}.json.gz",
        num_shards=1024,
        limit=4,
    )
