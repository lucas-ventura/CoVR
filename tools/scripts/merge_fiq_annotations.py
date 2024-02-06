
import json
from collections import defaultdict
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


def main(ann_dir: str):
    dress_types = ["dress", "shirt", "toptee"]
    splits = ["train", "val"]

    ann_dir = Path(ann_dir)

    all_data_cap = defaultdict(list)
    all_data_tar = defaultdict(list)
    for split in splits:
        for dress_type in dress_types:
            all_data_cap[split].extend(json_load(ann_dir / f"cap.{dress_type}.{split}.json"))
            all_data_tar[split].extend(json_load(ann_dir / f"split.{dress_type}.{split}.json"))
        
        json_dump(all_data_cap[split], ann_dir / f"cap.all.{split}.json")
        json_dump(all_data_tar[split], ann_dir / f"split.all.{split}.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--fiq_dir", default="annotation/fashion-iq", help="Fashion IQ directory.")
    args = parser.parse_args()

    main(args.fiq_dir)

