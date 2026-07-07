import glob
import os

import cv2
import numpy as np
import tqdm


def average_frame_count(input_file_path, min_frames=150):
    input_path = f"{input_file_path}/*.mp4"
    video_files = glob.glob(input_path)
    frame_count = []
    video_list = []
    short_frame = []
    short_frame_count = []

    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if count < min_frames:
            short_frame.append(video_file)
            short_frame_count.append(count)
            continue

        video_list.append(video_file)
        frame_count.append(count)

    print("Total number of videos:", len(frame_count))
    print("Average frame per video:", np.mean(frame_count) if frame_count else 0)
    print("Short frame video:", len(short_frame))
    return video_list


def frame_extract(path):
    video = cv2.VideoCapture(path)
    success = True
    while success:
        success, image = video.read()
        if success:
            yield image
    video.release()


def create_face_videos(path_list, out_dir, mtcnn, output_size=224):
    already_present_count = glob.glob(out_dir + "/*.mp4")
    print("No of videos already present:", len(already_present_count))
    os.makedirs(out_dir, exist_ok=True)

    for path in tqdm.tqdm(path_list):
        out_path = os.path.join(out_dir, os.path.basename(path))
        if os.path.exists(out_path):
            print("File already exists:", out_path)
            continue

        out = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            30,
            (output_size, output_size),
        )

        for frame in frame_extract(path):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            box, _ = mtcnn.detect(rgb_frame)

            if box is not None:
                x1, y1, x2, y2 = map(int, box[0])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                face = frame[y1:y2, x1:x2]
                if face.size != 0:
                    resized_face = cv2.resize(face, (output_size, output_size))
                    out.write(resized_face)

        out.release()
