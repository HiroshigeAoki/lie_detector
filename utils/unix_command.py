import os
from pathlib import Path


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def mkdirs(base_path: Path):
    r"""
    Make hierarchical directory.
    :param base_path:
    :return:
    """
    path = Path("./")
    for dir_ in str(base_path).split("/"):
        path = path / dir_
        if not Path.exists(path):
            mkdir(path)
