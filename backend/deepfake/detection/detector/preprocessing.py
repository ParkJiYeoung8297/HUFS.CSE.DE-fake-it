import os
import cv2
import json
import torch
from facenet_pytorch import MTCNN

CPU_THREAD_LIMIT = int(os.environ.get("DEFAKE_CPU_THREADS", "4"))
ROI_METADATA_FILENAME = "roi_metadata.json"
FRAME_SAMPLE_STRIDE = max(1, int(os.environ.get("DEFAKE_FRAME_SAMPLE_STRIDE", "5")))

torch.set_num_threads(CPU_THREAD_LIMIT)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
cv2.setNumThreads(1)

def frame_extract(path):
    cap = cv2.VideoCapture(path)
    while True:
        success, frame = cap.read()
        if not success:
            break
        yield frame
    cap.release()


def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def _box_from_center(center_x, center_y, half_width, half_height, image_size=224):
    x1 = int(round(center_x - half_width))
    y1 = int(round(center_y - half_height))
    x2 = int(round(center_x + half_width))
    y2 = int(round(center_y + half_height))
    return [
        _clamp(x1, 0, image_size - 1),
        _clamp(y1, 0, image_size - 1),
        _clamp(x2, 1, image_size),
        _clamp(y2, 1, image_size),
    ]


def _default_roi_bboxes(output_size=224):
    return {
        "Jawline": [0, int(output_size * 0.58), output_size, output_size],
        "Left Eye": [int(output_size * 0.22), int(output_size * 0.30), int(output_size * 0.45), int(output_size * 0.42)],
        "Right Eye": [int(output_size * 0.55), int(output_size * 0.30), int(output_size * 0.78), int(output_size * 0.42)],
        "Left Eyebrow": [int(output_size * 0.20), int(output_size * 0.21), int(output_size * 0.45), int(output_size * 0.31)],
        "Right Eyebrow": [int(output_size * 0.55), int(output_size * 0.21), int(output_size * 0.80), int(output_size * 0.31)],
        "Nose": [int(output_size * 0.42), int(output_size * 0.40), int(output_size * 0.58), int(output_size * 0.63)],
        "Mouth": [int(output_size * 0.32), int(output_size * 0.66), int(output_size * 0.68), int(output_size * 0.80)],
    }


def _build_roi_bboxes_from_mtcnn(face_box, landmarks, output_size=224):
    x1, y1, x2, y2 = [float(v) for v in face_box]
    face_w = max(x2 - x1, 1.0)
    face_h = max(y2 - y1, 1.0)

    def scale_point(point):
        px, py = point
        return (
            _clamp((float(px) - x1) / face_w * output_size, 0, output_size),
            _clamp((float(py) - y1) / face_h * output_size, 0, output_size),
        )

    left_eye, right_eye, nose, mouth_left, mouth_right = [
        scale_point(point) for point in landmarks
    ]

    eye_distance = max(abs(right_eye[0] - left_eye[0]), output_size * 0.18)
    eye_half_w = eye_distance * 0.22
    eye_half_h = output_size * 0.045
    eyebrow_offset = output_size * 0.075

    mouth_center_x = (mouth_left[0] + mouth_right[0]) / 2
    mouth_center_y = (mouth_left[1] + mouth_right[1]) / 2
    mouth_half_w = max(abs(mouth_right[0] - mouth_left[0]) * 0.75, output_size * 0.11)

    return {
        "Jawline": [0, int(output_size * 0.58), output_size, output_size],
        "Left Eye": _box_from_center(left_eye[0], left_eye[1], eye_half_w, eye_half_h, output_size),
        "Right Eye": _box_from_center(right_eye[0], right_eye[1], eye_half_w, eye_half_h, output_size),
        "Left Eyebrow": _box_from_center(left_eye[0], left_eye[1] - eyebrow_offset, eye_half_w * 1.1, eye_half_h, output_size),
        "Right Eyebrow": _box_from_center(right_eye[0], right_eye[1] - eyebrow_offset, eye_half_w * 1.1, eye_half_h, output_size),
        "Nose": _box_from_center(nose[0], nose[1], output_size * 0.07, output_size * 0.10, output_size),
        "Mouth": _box_from_center(mouth_center_x, mouth_center_y, mouth_half_w, output_size * 0.07, output_size),
    }


def process_single_video(video_path, output_path, filename):
    os.makedirs(output_path, exist_ok=True)
    output_video_path = os.path.join(output_path, filename)
    metadata_path = os.path.join(output_path, ROI_METADATA_FILENAME)

    device = torch.device("mps") if torch.backends.mps.is_available() else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}")

    mtcnn = MTCNN(keep_all=False, device=torch.device('cpu'))

    output_fps = max(1, round(30 / FRAME_SAMPLE_STRIDE))

    # 저장용 비디오 객체 (224x224 크기)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), output_fps, (224,224))
    metadata = {
        "source_video": os.path.basename(video_path),
        "preprocessed_video": filename,
        "output_size": [224, 224],
        "frame_sample_stride": FRAME_SAMPLE_STRIDE,
        "output_fps": output_fps,
        "frames": [],
    }
    source_frame_idx = 0
    written_frame_idx = 0

    for frame in frame_extract(video_path):
        try:
            if source_frame_idx % FRAME_SAMPLE_STRIDE != 0:
                continue

            # RGB로 변환
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _, landmarks = mtcnn.detect(rgb, landmarks=True)

            if boxes is not None:
                x1, y1, x2, y2 = map(int, boxes[0])  # 첫 번째 얼굴만 사용
                h, w, _ = frame.shape
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                face = frame[y1:y2, x1:x2]
                if face.size != 0:
                    resized = cv2.resize(face, (224,224))
                    out.write(resized)

                    roi_bboxes = (
                        _build_roi_bboxes_from_mtcnn([x1, y1, x2, y2], landmarks[0])
                        if landmarks is not None
                        else _default_roi_bboxes()
                    )
                    metadata["frames"].append({
                        "source_frame_idx": source_frame_idx,
                        "preprocessed_frame_idx": written_frame_idx,
                        "face_bbox": [x1, y1, x2, y2],
                        "roi_bboxes": roi_bboxes,
                    })

                    written_frame_idx += 1
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue
        finally:
            source_frame_idx += 1

    out.release()
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False)

    print("Saved:", output_video_path)
    print("Saved ROI metadata:", metadata_path)
    return output_video_path
