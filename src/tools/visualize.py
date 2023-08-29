from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image


def get_video_frames(video_pth, frames_video=15):
    import cv2

    assert Path(video_pth).exists(), f"Video {video_pth} does not exist"

    video_pth = str(video_pth)

    # use OpenCV to read the video
    cap = cv2.VideoCapture(video_pth)

    # get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idxs = sample_frames(total_frames, n_frames=frames_video)

    frames = []
    f_idxs = []
    for frame_idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            print(f"Video {video_pth} is corrupted")
            frames = [
                Image.fromarray(np.zeros((384, 384, 3)).astype(np.uint8))
            ] * frames_video
            f_idxs = [-1] * frames_video
            return frames, f_idxs

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
        f_idxs.append(frame_idx)

    # pad frames to have the same number of frames
    n_frames = len(frames)
    if n_frames < frames_video:
        frames += [Image.fromarray(np.zeros((384, 384, 3)).astype(np.uint8))] * (
            frames_video - n_frames
        )

    # Add -1 to f_idxs for the remaining frames
    f_idxs += [-1] * (frames_video - len(f_idxs))

    return frames, f_idxs


def sample_frames(vlen, n_frames=15):
    acc_samples = min(vlen, n_frames)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))

    frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

    return frame_idxs


def concat_h_imgs(im_list, resample=Image.Resampling.BICUBIC):
    min_height = min(im.height for im in im_list)
    im_list_resize = [
        im.resize(
            (int(im.width * min_height / im.height), min_height), resample=resample
        )
        for im in im_list
    ]
    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new("RGB", (total_width, min_height))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    return dst


def extract_frames(url, n_frames=10):
    import urllib.request

    import cv2
    import numpy as np
    from PIL import Image

    # Download the video from the URL
    resp = urllib.request.urlopen(url)
    video = resp.read()

    # Load the video using OpenCV
    video = cv2.VideoCapture(url)

    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample `n_frames` frames from the video
    frames_idxs = sample_frames(total_frames, n_frames=n_frames)

    frames = []
    for frame_number in frames_idxs:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()
        if ret:
            # Convert the color channels from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the NumPy array to a PIL Image object
            pil_image = Image.fromarray(np.uint8(frame))
            frames.append(pil_image)
    return frames


def visualize_url_video(url, n_frames=10):
    frames = extract_frames(url, n_frames=n_frames)
    if n_frames == 1:
        return frames[0]
    return concat_h_imgs(frames)


def visualize_pth_video(video_pth, n_frames=10):
    frames, _ = get_video_frames(video_pth, frames_video=n_frames)
    if n_frames == 1:
        return frames[0]
    return concat_h_imgs(frames)


def visualize_video(video, n_frames=10):
    if is_url(video):
        return visualize_url_video(video, n_frames=n_frames)
    return visualize_pth_video(video, n_frames=n_frames)


def is_url(url_or_filename):
    parsed = urlparse(str(url_or_filename))
    return parsed.scheme in ("http", "https")
