from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from src.data.utils import pre_caption
from src.tools.files import read_txt

normalize = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
)


class ImageDataset(Dataset):
    def __init__(
        self,
        image_dir,
        save_dir=None,
        todo_ids=None,
        pixel_size=384,
    ):
        self.image_dir = Path(image_dir)

        if isinstance(todo_ids, (str, Path)):
            todo_ids = get_ids(todo_ids)

        found_pths = list(self.image_dir.glob("*.jpg")) + list(
            self.image_dir.glob("*.png")
        )
        found_ids = [pth.stem for pth in found_pths]

        if todo_ids is not None:
            video_ids = set(todo_ids) & set(found_ids)
        else:
            video_ids = found_ids

        self.id2pth = {
            img_pth.stem: img_pth for img_pth in found_pths if img_pth.stem in video_ids
        }
        self.video_ids = list(self.id2pth.keys())
        self.video_ids.sort()

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.pixel_size = pixel_size

        if save_dir is not None:
            save_dir = Path(save_dir)
            done_paths = list(save_dir.glob("*.pth"))
            done_paths = {p.stem for p in done_paths}
            print(f"video_ids: {len(self.video_ids)} - {len(done_paths)} = ", end="")
            self.video_ids = list(set(self.video_ids) - done_paths)
            print(len(self.video_ids))
            self.video_ids.sort()
            if len(self.video_ids) == 0:
                print("All videos are done")
                exit()

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]

        img_pth = self.id2pth[video_id]
        try:
            img = Image.open(img_pth).convert("RGB")
            img = self.transform(img)
        except:  # noqa: E722
            print(f"Image {img_pth} is corrupted")
            img = torch.zeros((3, self.pixel_size, self.pixel_size))
            video_id = "delete"

        return img, video_id


class VideoDataset(Dataset):
    def __init__(
        self,
        video_dir,
        todo_ids=None,
        shard_id=0,
        num_shards=1,
        frames_video=15,
        extension="mp4",
        save_dir=None,
        image_size=384,
    ):
        self.video_dir = Path(video_dir)

        if isinstance(todo_ids, (str, Path)):
            todo_ids = get_ids(todo_ids)

        found_paths = list(video_dir.glob(f"*/*.{extension}"))
        if todo_ids is not None:
            video_paths = [video_dir / f"{v}.{extension}" for v in todo_ids]
            video_paths = list(set(video_paths) & set(found_paths))
        else:
            video_paths = found_paths
        video_paths.sort()
        self.id2path = {pth.parent.name + "/" + pth.stem: pth for pth in video_paths}
        self.video_ids = list(self.id2path.keys())
        self.video_ids.sort()

        if save_dir is not None:
            save_dir = Path(save_dir)
            done_paths = list(save_dir.glob("*/*.pth"))
            done_paths = {p.parent.name + "/" + p.stem for p in done_paths}
            print(f"video_ids: {len(self.video_ids)} - {len(done_paths)} = ", end="")
            self.video_ids = list(set(self.video_ids) - done_paths)
            print(len(self.video_ids))
            self.video_ids.sort()
            if len(self.video_ids) == 0:
                print("All videos are done")
                exit()

        assert len(self.video_ids) > 0, "video_ids is empty"

        # shard the dataset
        n_videos = len(self.video_ids)
        self.video_ids = self.video_ids[
            shard_id * n_videos // num_shards : (shard_id + 1) * n_videos // num_shards
        ]

        self.frames_video = frames_video
        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        video_path = self.id2path[video_id]
        frames, f_idxs = get_video_frames(
            video_path, self.frames_video, self.image_size
        )
        frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames, dim=0)
        f_idxs = torch.tensor(f_idxs)

        return video_id, f_idxs, frames


class TextDataset(Dataset):
    def __init__(
        self,
        csv_path,
        max_words=30,
    ):
        self.df = pd.read_csv(csv_path)
        self.texts = list(set(self.df["edit"].unique().tolist()))
        self.texts.sort()
        self.max_words = max_words

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        txt = self.texts[index]
        txt = pre_caption(txt, self.max_words)

        return txt


def get_video_frames(video_pth, frames_video=15, image_size=384):
    import cv2

    video_pth = str(video_pth)

    # use OpenCV to read the video
    cap = cv2.VideoCapture(video_pth)

    # get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idxs = sample_frames(total_frames, frames_video)

    frames = []
    f_idxs = []
    for frame_idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            print(f"Video {video_pth} is corrupted")
            frames = [
                Image.fromarray(np.zeros((image_size, image_size, 3)).astype(np.uint8))
            ] * frames_video
            f_idxs = [-1] * frames_video
            return frames, f_idxs

        frames.append(Image.fromarray(frame))
        f_idxs.append(frame_idx)

    # pad frames to have the same number of frames
    n_frames = len(frames)
    if n_frames < frames_video:
        frames += [
            Image.fromarray(np.zeros((image_size, image_size, 3)).astype(np.uint8))
        ] * (frames_video - n_frames)

    # Add -1 to f_idxs for the remaining frames
    f_idxs += [-1] * (frames_video - len(f_idxs))

    return frames, f_idxs


def sample_frames(vlen, frames_per_video=15):
    acc_samples = min(vlen, frames_per_video)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))

    frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

    return frame_idxs


def get_ids(path):
    suffix = Path(path).suffix
    if suffix == ".csv":
        df = pd.read_csv(path)
        ids = set(df["pth2"].tolist())
        ids = list(ids)
        ids.sort()
    else:
        ids = read_txt(path)

    return ids
