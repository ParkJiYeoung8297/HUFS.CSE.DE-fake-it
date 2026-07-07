from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms


# =========================
# User configuration
# =========================
# Edit these values before running the script.
INPUT_DIR = "/path/to/Dataset/ff++"
OUTPUT_DIR = None
FRAMES = 150
SAMPLE_COUNT = 20


def frame_extract(path):
    video = cv2.VideoCapture(str(path))
    success = True
    while success:
        success, image = video.read()
        if success:
            yield image
    video.release()


def validate_video(video_path, transform, sample_count=20):
    frames = []
    for frame in frame_extract(video_path):
        frames.append(transform(frame))
        if len(frames) == sample_count:
            break

    if len(frames) < sample_count:
        raise ValueError(f"not enough readable frames: {len(frames)}")

    return torch.stack(frames)


def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    video_files = sorted(input_dir.rglob("*.mp4"))
    valid_videos = []
    short_videos = []
    corrupted_videos = []
    frame_counts = []

    print("Total videos:", len(video_files))
    for index, video_file in enumerate(video_files, start=1):
        try:
            validate_video(video_file, transform, SAMPLE_COUNT)
            cap = cv2.VideoCapture(str(video_file))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if frame_count < FRAMES:
                short_videos.append(str(video_file))
            else:
                valid_videos.append(str(video_file))
                frame_counts.append(frame_count)
        except Exception as exc:
            corrupted_videos.append({"video_file": str(video_file), "error": str(exc)})
            print(f"Invalid video ({index}/{len(video_files)}): {video_file} - {exc}")

    pd.DataFrame(valid_videos, columns=["video_file"]).to_excel(
        output_dir / "validate_video_files.xlsx",
        index=False,
    )
    pd.DataFrame(short_videos, columns=["video_file"]).to_excel(
        output_dir / "delete_video_files.xlsx",
        index=False,
    )
    pd.DataFrame(corrupted_videos).to_excel(
        output_dir / "corrupted_video_files.xlsx",
        index=False,
    )

    print("Validated videos:", len(valid_videos))
    print("Short videos:", len(short_videos))
    print("Corrupted videos:", len(corrupted_videos))
    if frame_counts:
        print("Average frame count:", float(np.mean(frame_counts)))


if __name__ == "__main__":
    main()
