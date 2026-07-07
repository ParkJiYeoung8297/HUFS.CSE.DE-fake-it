from pathlib import Path

import cv2


# =========================
# User configuration
# =========================
# Edit these values before running the script.
INPUT_DIR = "/path/to/input/videos"
OUTPUT_DIR = "/path/to/output/videos"
FRAMES = 150


def save_first_frames(video_path, output_video_path, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            break
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    return output_video_path, frame_count


def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(input_dir.rglob("*.mp4"))
    print("Total videos:", len(video_files))

    for video_path in video_files:
        relative_path = video_path.relative_to(input_dir)
        output_video_path = output_dir / relative_path
        output_video_path, frame_count = save_first_frames(video_path, output_video_path, FRAMES)
        print(f"Saved {frame_count} frames: {video_path.name} -> {output_video_path}")

    print("Finished saving videos.")


if __name__ == "__main__":
    main()
