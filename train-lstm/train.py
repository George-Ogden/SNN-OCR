import os

from torchdata.datapipes.iter import FileLister, FileOpener

from config import data_root

data_path = os.path.join(os.path.dirname(__file__), data_root)
dp = FileLister(root=data_path).filter(lambda fname: fname.endswith(".txt"))
dp = FileOpener(dp)

for line in dp.readlines(strip_newline=False, return_path=False).shuffle().sharding_filter():
    print(line)
    break
