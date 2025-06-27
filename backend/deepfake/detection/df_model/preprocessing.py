import os
import cv2
import torch
from facenet_pytorch import MTCNN

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
    print(f"Using device: {device}")

    mtcnn = MTCNN(keep_all=False, device=torch.device('cpu'))

    # 저장용 비디오 객체 (112x112 크기)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (224,224))

    for frame in frame_extract(video_path):
        try:
            # RGB로 변환
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
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

    out.release()
    print("Saved:", output_video_path)
    return output_video_path
