import logging
import os
import cv2
import torch
from facenet_pytorch import MTCNN


logger = logging.getLogger(__name__)

CPU_THREAD_LIMIT = int(os.environ.get("DEFAKE_CPU_THREADS", "4"))
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


def process_single_video(video_path, output_path, filename):
    os.makedirs(output_path, exist_ok=True)
    output_video_path = os.path.join(output_path, filename)

    device = torch.device("mps") if torch.backends.mps.is_available() else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    logger.debug("Using device for preprocessing: %s", device)

    mtcnn = MTCNN(keep_all=False, device=torch.device('cpu'))

    output_fps = max(1, round(30 / FRAME_SAMPLE_STRIDE))

    # 저장용 비디오 객체 (224x224 크기)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), output_fps, (224,224))
    source_frame_idx = 0

    for frame in frame_extract(video_path):
        try:
            if source_frame_idx % FRAME_SAMPLE_STRIDE != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(rgb)

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
        except Exception as exc:
            logger.warning("Error processing frame %s: %s", source_frame_idx, exc)
            continue
        finally:
            source_frame_idx += 1

    out.release()

    logger.debug("Saved preprocessed video: %s", output_video_path)
    return output_video_path
