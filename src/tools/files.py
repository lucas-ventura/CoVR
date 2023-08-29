import json
from pathlib import Path
from typing import Union


def json_load(json_pth: Union[Path, str]):
    if not isinstance(json_pth, str):
        json_pth = str(json_pth)
    with open(json_pth) as f:
        data = json.load(f)
    return data


def json_dump(data, json_pth: Union[Path, str]):
    if not isinstance(json_pth, str):
        json_pth = str(json_pth)
    with open(json_pth, "w") as f:
        json.dump(data, f, indent=2)


def read_txt(txt_pth: Union[Path, str]) -> list:
    with open(txt_pth) as f:
        lines = f.read().split("\n")
    return lines[:-1]


def write_txt(data, txt_pth: Union[Path, str]):
    with open(txt_pth, "w") as f:
        for item in data:
            f.write("%s\n" % item)
