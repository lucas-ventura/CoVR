# Code based from https://github.com/m-bain/webvid/blob/main/download.py

import numpy as np
import argparse
import requests
import concurrent.futures
from mpi4py import MPI
import warnings 
from pathlib import Path
from tqdm.auto import tqdm

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

def request_save(url, save_fp):
    img_data = requests.get(url, timeout=5).content
    with open(save_fp, 'wb') as handler:
        handler.write(img_data)

def get_urls(split):
    if split == 'train':
        url = "http://imagine.enpc.fr/~ventural/covr/dataset/webvid2m-covr_paths-train.json"
    elif split == 'val':
        url = "http://imagine.enpc.fr/~ventural/covr/dataset/webvid8m-covr_paths-val.json"
    elif split == 'test':
        url = "http://imagine.enpc.fr/~ventural/covr/dataset/webvid8m-covr_paths-test.json"
    else:
        raise ValueError("Split must be one of train, val, or test")

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        path2url = response.json()  # Parse JSON data from the response
        return path2url

    except requests.exceptions.RequestException as e:
        print(f"Error fetching JSON: {e}")

def main(args):
    if args.split == 'train':
        video_dir = Path(f"{args.data_dir}/WebVid/2M/train")
    elif args.split in ['val', 'test']:
        video_dir = Path(f"{args.data_dir}/WebVid/8M/train")
    if RANK == 0:
        video_dir.mkdir(parents=True, exist_ok=True)
    COMM.barrier()

    path2url = get_urls(args.split)
    paths = set(path2url.keys())

    # Remove paths that have already been downloaded
    found_paths = list(video_dir.glob('*/*.mp4'))
    found_paths = {str(p.relative_to(video_dir)) for p in found_paths}
    paths = list(paths - found_paths)
    paths.sort()

    # Split paths into partitions
    paths = np.array_split(paths, args.partitions)[args.part]

    for path in paths:
        vid_path = video_dir / path
        vid_dir = vid_path.parent
        vid_dir.mkdir(exist_ok=True)

    path2url = {path: path2url[path] for path in paths}
    
    # split into batches of 1000
    for i in tqdm(range(0, len(path2url), 1000)):
        path2url_batch = {path: path2url[path] for path in list(path2url.keys())[i:i+1000]}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.processes) as executor:
            {executor.submit(request_save, url, video_dir / path) for path, url in path2url_batch.items()}




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shutter Image/Video Downloader')
    parser.add_argument('--partitions', type=int, default=1,
                        help='Number of partitions to split the dataset into, to run multiple jobs in parallel')
    parser.add_argument('--part', type=int, default=0,
                        help='Partition number to download where 0 <= part < partitions')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='Directory where webvid data is stored.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Which split to download')
    parser.add_argument('--processes', type=int, default=8)
    args = parser.parse_args()

    if SIZE > 1:
        warnings.warn("Overriding --part with MPI rank number")
        args.part = RANK

    if args.part >= args.partitions:
        raise ValueError("Part idx must be less than number of partitions")
    main(args)