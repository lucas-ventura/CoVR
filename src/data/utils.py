import re
from typing import Union

import torch


def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        " ",
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    # truncate caption
    caption_words = caption.split(" ")
    if len(caption_words) > max_words:
        caption = " ".join(caption_words[:max_words])

    return caption


def id2int(data, sub=""):
    if isinstance(data, list):
        return [remove_non_digits(d, sub) for d in data]
    else:
        return remove_non_digits(data, sub)


def remove_non_digits(string, sub: str = ""):
    return int(re.sub(r"\D", sub, string))


def get_middle_frame(reference_vid_pth):
    from pathlib import Path

    import cv2
    import numpy as np
    from PIL import Image

    reference_vid_pth = str(reference_vid_pth)

    if not Path(reference_vid_pth).exists():
        print(f"Video {reference_vid_pth} does not exist")
        return Image.fromarray(np.zeros((384, 384, 3)).astype(np.uint8))

    # use OpenCV to read the video
    cap = cv2.VideoCapture(reference_vid_pth)

    # get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # calculate the index of the middle frame
    middle_frame_index = total_frames // 2

    # set the current frame index to the middle frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)

    # read the middle frame
    ret, frame = cap.read()

    if not ret or frame is None:
        print(f"Video {reference_vid_pth} is corrupted")
        return Image.fromarray(np.zeros((384, 384, 3)).astype(np.uint8))

    # convert the frame from BGR to RGB using OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # create a PIL Image object from the middle frame
    pil_image = Image.fromarray(frame)

    return pil_image


def get_random_frame(reference_vid_pth):
    from pathlib import Path

    import cv2
    import numpy as np
    from PIL import Image

    reference_vid_pth = str(reference_vid_pth)

    if not Path(reference_vid_pth).exists():
        print(f"Video {reference_vid_pth} does not exist")
        return Image.fromarray(np.zeros((384, 384, 3)).astype(np.uint8))

    # use OpenCV to read the video
    cap = cv2.VideoCapture(reference_vid_pth)

    # get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # calculate the index of random frame
    random_frame_index = np.random.randint(0, total_frames)

    # set the current frame index to the random frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)

    # read the frame
    ret, frame = cap.read()

    if not ret or frame is None:
        print(f"Video {reference_vid_pth} is corrupted")
        return Image.fromarray(np.zeros((384, 384, 3)).astype(np.uint8))

    # convert the frame from BGR to RGB using OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # create a PIL Image object from the middle frame
    pil_image = Image.fromarray(frame)

    return pil_image


def sample_frames(frames_videos, vlen):
    import numpy as np

    acc_samples = min(frames_videos, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))

    frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

    return frame_idxs


class FrameLoader:
    def __init__(self, transform, frames_video=1, method="middle"):
        self.transform = transform
        self.method = method

        if method == "middle":
            self.get_frame = get_middle_frame
            assert frames_video == 1, "frames_video must be 1 for middle frame method"
        elif method == "random":
            self.get_frame = get_random_frame
            assert frames_video == 1, "frames_video must be 1 for random frame method"
        elif method == "sample":
            assert frames_video > 1, "frames_video must be > 1 for sample frame method"
            self.frames_video = frames_video
        else:
            raise ValueError(f"Invalid method: {method}")

    def __call__(self, video_pth: str):
        if self.method == "sample":
            frames = self.get_video_frames(video_pth, 0.0, None)
            return torch.stack(frames)
        else:
            return self.transform(self.get_frame(video_pth))

    def get_video_frames(
        self,
        video_pth: str,
        start_time: float = 0.0,
        end_time: Union[float, None] = None,
    ) -> list:
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(video_pth)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if end_time is not None:
            start_frame = int(fps * start_time)
            end_frame = int(fps * end_time)
            vlen = end_frame - start_frame
        else:
            start_frame = 0
            vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idxs = sample_frames(self.frames_video, vlen)
        frame_idxs = [frame_idx + start_frame for frame_idx in frame_idxs]
        if self.frames_video != len(frame_idxs):
            frame_idxs = (frame_idxs * self.frames_video)[: self.frames_video]
            print(f"Video {video_pth} has less than {self.frames_video} frames")

        frames = []
        for index in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb).convert("RGB"))

        cap.release()

        if len(frames) > 0:
            video_data = [self.transform(frame) for frame in frames]
            return video_data
        else:
            raise ValueError(f"video path: {video_pth} error.")
