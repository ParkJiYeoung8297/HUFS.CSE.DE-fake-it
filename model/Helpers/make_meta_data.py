import glob
import os
from pathlib import Path

import cv2
import pandas as pd


# =========================
# User configuration
# =========================
# Edit these values before running the script.
INPUT_GLOB = "/path/to/Dataset/ff++/**/*.mp4"
BASE_DIR = "/path/to/Dataset"
OUTPUT_DIR = "/path/to/Dataset/ff++"
CSV_NAME = "Global_metadata.csv"
EXCEL_NAME = "global_meta_data.xlsx"
CSV_HEADER = False


METHOD_LABELS = {
    "original": "original",
    "faceswap": "FaceSwap",
    "faceshifter": "FaceShifter",
    "face2face": "Face2Face",
    "neuraltextures": "NeuralTextures",
    "deepfakes": "Deepfakes",
}


def infer_label(relative_path):
    path = relative_path.lower()
    if "real" in path:
        return "REAL"
    if "fake" in path:
        return "FAKE"
    return "unknown"


def infer_split(relative_path):
    path = relative_path.lower()
    for split in ("train", "validation", "val", "test"):
        if f"/{split}/" in path:
            return "validation" if split == "val" else split
    return "unknown"


def infer_dataset(relative_path):
    path = relative_path.lower()
    if "ff++" in path or "faceforensics" in path:
        return "ff++"
    if "celeb" in path:
        return "celeb-df"
    if "deeperforensics" in path:
        return "deeperforensics"
    if "dfdc" in path:
        return "dfdc"
    return "unknown"


def infer_method(relative_path):
    path = relative_path.lower()
    for token, label in METHOD_LABELS.items():
        if token in path:
            return label
    return "unknown"


def collect_metadata(input_glob, base_dir):
    rows = []
    for video_file in sorted(glob.glob(input_glob, recursive=True)):
        video_path = Path(video_file)
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        relative_path = os.path.relpath(video_path, base_dir).replace("\\", "/")
        rows.append(
            {
                "file_name": video_path.name,
                "folder_path": relative_path,
                "label": infer_label(relative_path),
                "split": infer_split(relative_path),
                "dataset": infer_dataset(relative_path),
                "method": infer_method(relative_path),
                "frame": frame_count,
            }
        )
    return rows


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_metadata(INPUT_GLOB, BASE_DIR)
    full_df = pd.DataFrame(rows)
    compact_df = full_df[["file_name", "label", "method"]]

    excel_path = output_dir / EXCEL_NAME
    csv_path = output_dir / CSV_NAME

    full_df.to_excel(excel_path, index=False, engine="openpyxl")
    compact_df.to_csv(
        csv_path,
        index=False,
        header=CSV_HEADER,
        encoding="utf-8-sig",
    )

    print(f"Metadata rows: {len(full_df)}")
    print(f"Excel metadata saved to: {excel_path}")
    print(f"Training CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
