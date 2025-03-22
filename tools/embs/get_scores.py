from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def main(data_csv, embs_dir, num_shards=0, shard_id=1):
    df = pd.read_csv(data_csv)
    txt2_embs = torch.load(embs_dir / f"edit-{data_csv.stem}.pth")
    txt2emb = dict(zip(txt2_embs["edits"], txt2_embs["feats"]))

    df = df.iloc[shard_id::num_shards]

    scores = []
    for row in tqdm(df.itertuples(), total=len(df)):
        edit = row.edit
        pth2 = embs_dir / f"{row.pth2}.pth"
        if pth2.exists():
            txt_emb = txt2emb[edit]
            vid_emb = torch.load(pth2)

            row_scores = torch.einsum("fe,e->f", vid_emb, txt_emb).tolist()
            row_scores = [round(score, 4) for score in row_scores]
            scores.append(row_scores)

        else:
            scores.append([])

    df["scores"] = scores
    shards_id = f"-{shard_id}-{num_shards}" if num_shards > 1 else ""
    out_data_csv = data_csv.parent / f"{data_csv.stem}_qs{shards_id}.csv"
    df.to_csv(out_data_csv, index=False)

    # check if all shards have been processed
    if num_shards > 1:
        out_data_csvs = list(data_csv.parent.glob(f"{data_csv.stem}_qs-*.csv"))
        if len(out_data_csvs) == num_shards + 1:
            out_data_csvs.sort(key=lambda x: int(x.stem.split("-")[-2]))
            df = pd.concat([pd.read_csv(csv) for csv in out_data_csvs])
            df = df.drop_duplicates(subset=["pth1", "pth2"])
            df.to(data_csv.parent / f"{data_csv.stem}_qs.csv", index=False)
            for csv in out_data_csvs:
                csv.unlink()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data_csv", type=Path)
    parser.add_argument("save_dir", type=Path)
    parser.add_argument("-n", "--num_shards", type=int, default=1)
    parser.add_argument("-i", "--shard_id", type=int, default=0)
    args = parser.parse_args()

    main(args.data_csv, args.save_dir, args.num_shards, args.shard_id)
