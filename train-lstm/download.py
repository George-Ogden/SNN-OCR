import os
import subprocess
import sys

from config import data_root

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

data_path = os.path.join(os.path.dirname(__file__), data_root)


def download(url: str) -> str:
    filename = os.path.basename(url)
    os.makedirs(data_path, exist_ok=True)
    path = os.path.join(data_path, filename)
    if not os.path.exists(path):
        subprocess.run(
            f"curl -L {url} -o {path}", shell=True, check=True, capture_output=True, text=True
        )
    return filename


def unzip(filename: str):
    path = os.path.join(data_path, filename)
    result = subprocess.run(
        f"tar -xvf {path} -C {data_path}", shell=True, check=True, capture_output=True, text=True
    )
    print(result.stdout)
    os.unlink(path)
    with open(path, "x"):
        ...


if __name__ == "__main__":
    filename = download(
        "https://storage.googleapis.com/huggingface-nlp/datasets/bookcorpus/bookcorpus.tar.bz2"
    )
    unzip(filename)
