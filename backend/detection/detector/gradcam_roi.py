import cv2
import face_alignment
import logging
import torch
import torchvision.transforms as T
import numpy as np
import torch.nn.functional as F
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .model import Model


logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).resolve().parent


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )


def load_model(selected_model, checkpoint_name, target_device):
    model = Model(
        num_binary_classes=2,
        num_method_classes=7,
        model_name=selected_model
    ).to(target_device)
    model.load_state_dict(
        torch.load(CHECKPOINT_DIR / f"{checkpoint_name}.pt", map_location=target_device)
    )
    model.eval()
    return model


device = get_device()
logger.debug("Using device for Grad-CAM: %s", device)

CPU_THREAD_LIMIT = int(os.environ.get("DEFAKE_CPU_THREADS", "4"))
GRADCAM_PIPELINE_BUFFER = int(os.environ.get("DEFAKE_GRADCAM_PIPELINE_BUFFER", "2"))

torch.set_num_threads(CPU_THREAD_LIMIT)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
cv2.setNumThreads(1)


def compute_gradcam_binary(model, input_tensor, target_class=0, device_override=None):
    fmap = None
    grad = None
    target_device = device_override or device

    def fw_hook(module, inp, out):
        nonlocal fmap
        fmap = out.detach()

    def bw_hook(module, grad_in, grad_out):
        nonlocal grad
        grad = grad_out[0].detach()

    last_layer = model.model[-1]
    f = last_layer.register_forward_hook(fw_hook)
    b = last_layer.register_backward_hook(bw_hook)

    input_tensor = input_tensor.to(target_device).unsqueeze(0).unsqueeze(0).requires_grad_(True)
    _, binary_output, method_output = model(input_tensor)

    prob = F.softmax(binary_output, dim=1)[0, target_class].item()
    binary_pred = torch.argmax(binary_output, dim=1).item()
    method_pred = torch.argmax(method_output, dim=1).item()

    if binary_pred == 1 and method_pred == 0:
        cam = np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]))
        f.remove()
        b.remove()
        return cam, prob, binary_pred, method_pred

    target_class = 0
    model.zero_grad()
    binary_output[0, target_class].backward()

    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam = (weights * fmap).sum(dim=1, keepdim=True)
    cam = F.relu(cam).squeeze().cpu().numpy()

    if binary_pred == 0 and method_pred != 0:
        cam *= 1.5

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))

    f.remove()
    b.remove()
    return cam, prob, binary_pred, method_pred


def roi_activation(cam, bbox):
    x1, y1, x2, y2 = bbox
    patch = cam[y1:y2, x1:x2]
    if patch.size == 0:
        return -1
    mean_val = float(patch.mean())

    if np.isnan(mean_val):
        return -1
    return mean_val


def _convert_video(input_path, output_path, delete_source=False):
    try:
        subprocess.run([
            'ffmpeg', '-i', input_path,
            '-vcodec', 'libx264',
            '-acodec', 'aac',
            '-strict', 'experimental',
            '-y',
            output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        logger.debug("Video converted: %s", output_path)
        if delete_source and os.path.exists(input_path):
            os.remove(input_path)
    except subprocess.CalledProcessError as e:
        logger.warning("Error occurred during conversion: %s", e)


def convert_gradcam_outputs(video_dir):
    _convert_video(
        os.path.join(video_dir, "grad_cam_on_original.mp4"),
        os.path.join(video_dir, "converted_grad_cam_on_original.mp4"),
        delete_source=True
    )
    _convert_video(
        os.path.join(video_dir, "output_box_on_original.mp4"),
        os.path.join(video_dir, "converted_output_box_on_original.mp4"),
        delete_source=True
    )


def _compute_gradcam_for_rgb(model, transform, rgb):
    img = transform(rgb).to(device)
    return compute_gradcam_binary(model, img)


def get_bbox(points):
    x = points[:, 0]
    y = points[:, 1]
    return int(x.min()), int(y.min()), int(x.max()), int(y.max())


def _write_processed_frame(
    pending_item,
    facial_region,
    first_detection_count,
    second_detection_count,
    detection_probability,
    video_writer_box,
    video_writer_cam,
):
    frame = pending_item["frame"]
    bbox_map = pending_item["bbox_map"]
    cam, cam_score, binary_pred, method_pred = pending_item["gradcam_future"].result()
    cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]))

    scores = {region: roi_activation(cam, box) for region, box in bbox_map.items()}
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    first_activated_region = sorted_scores[0][0]
    second_activated_region = sorted_scores[1][0]

    f_x1, f_y1, f_x2, f_y2 = bbox_map[first_activated_region]
    s_x1, s_y1, s_x2, s_y2 = bbox_map[second_activated_region]

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    frame_with_box = frame.copy()
    overlay_with_box = overlay.copy()

    if scores[first_activated_region] > 0:
        cv2.rectangle(frame_with_box, (f_x1, f_y1), (f_x2, f_y2), (0, 255, 0), 2)
        cv2.rectangle(overlay_with_box, (f_x1, f_y1), (f_x2, f_y2), (0, 255, 0), 2)
    else:
        first_activated_region = "None"

    if scores[second_activated_region] > 0:
        cv2.rectangle(frame_with_box, (s_x1, s_y1), (s_x2, s_y2), (255, 0, 0), 2)
        cv2.rectangle(overlay_with_box, (s_x1, s_y1), (s_x2, s_y2), (255, 0, 0), 2)
    else:
        second_activated_region = "None"

    video_writer_box.write(frame_with_box)
    video_writer_cam.write(overlay_with_box)

    first_detection_count[first_activated_region] += 1
    second_detection_count[second_activated_region] += 1
    for key in facial_region:
        if key != "None" and scores[key] != -1:
            detection_probability[key] += cam_score * scores[key]


def calculate_roi_scores(video_dir, file_name, result, model):
    os.makedirs(video_dir, exist_ok=True)
    face_aligner = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device=str(device),
    )

    transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    facial_region=['Jawline', 'Left Eye', 'Right Eye', 'Left Eyebrow', 'Right Eyebrow', 'Nose', 'Mouth','None']
    first_detection_count = {key: 0 for key in facial_region}
    second_detection_count = {key: 0 for key in facial_region}
    detection_probability = {key: 0.0 for key in facial_region}

    video_path_file_name=os.path.join(video_dir,file_name)
    cap = cv2.VideoCapture(video_path_file_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_idx = 0
    processed_count = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 박스만 그린 영상
    video_writer_box = cv2.VideoWriter(
        os.path.join(video_dir, "output_box_on_original.mp4"),
        fourcc, fps, (width, height)
    )

    # Grad-CAM + 박스 영상
    video_writer_cam = cv2.VideoWriter(
        os.path.join(video_dir, f"grad_cam_on_original.mp4"),
        fourcc, fps, (width, height)
    )

    pending = []

    with ThreadPoolExecutor(max_workers=1) as gradcam_executor:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = face_aligner.get_landmarks(rgb)
            if not landmarks:
                frame_idx += 1
                continue

            lm = landmarks[0]
            bbox_map = {
                'Jawline': get_bbox(lm[0:17]),
                'Left Eye': get_bbox(lm[36:42]),
                'Right Eye': get_bbox(lm[42:48]),
                'Left Eyebrow': get_bbox(lm[17:22]),
                'Right Eyebrow': get_bbox(lm[22:27]),
                'Nose': get_bbox(lm[27:36]),
                'Mouth': get_bbox(lm[48:68]),
            }
            pending.append({
                "frame": frame.copy(),
                "bbox_map": bbox_map,
                "gradcam_future": gradcam_executor.submit(
                    _compute_gradcam_for_rgb,
                    model,
                    transform,
                    rgb,
                ),
            })

            if len(pending) >= GRADCAM_PIPELINE_BUFFER:
                _write_processed_frame(
                    pending.pop(0),
                    facial_region,
                    first_detection_count,
                    second_detection_count,
                    detection_probability,
                    video_writer_box,
                    video_writer_cam,
                )
                processed_count += 1

            frame_idx += 1

        while pending:
            _write_processed_frame(
                pending.pop(0),
                facial_region,
                first_detection_count,
                second_detection_count,
                detection_probability,
                video_writer_box,
                video_writer_cam,
            )
            processed_count += 1

    cap.release()
    video_writer_box.release()
    video_writer_cam.release()

    denominator = processed_count or 1
    first_detection_rate = {key: round((value / denominator)*100, 2) for key, value in first_detection_count.items()}
    second_detection_rate = {key: round((value / denominator)*100, 2) for key, value in second_detection_count.items()}
    raw_detection_probability = {key: round(value, 4) for key, value in detection_probability.items()}
    probability_total = sum(detection_probability.values())
    if probability_total:
        detection_probability = {key: round((value / probability_total) * 100, 2) for key, value in detection_probability.items()}
    else:
        detection_probability = {key: 0.0 for key in detection_probability}

    roi_analyze_result = {
    "video_name": file_name,
    "binary_pred":result["Prediction"],
    "final_probability":result["Probability"],
    "method_pred":result["Method"],
    "facial_region": facial_region,
    "first_detection_count": first_detection_count,
    "second_detection_count": second_detection_count,
    "first_detection_rate": first_detection_rate,
    "second_detection_rate": second_detection_rate,
    "raw_detection_probability": raw_detection_probability,
    "detection_probability": detection_probability
    }

    table_data = []
    for region in facial_region:
        row = {
            "region": region,
            "first_count": f"{first_detection_count.get(region, '0')} ({first_detection_rate.get(region, '0.00')}%)",
            "second_count": f"{second_detection_count.get(region, '0')} ({second_detection_rate.get(region, '0.00')}%)",
            "confidence": "-" if detection_probability.get(region) is None else f"{detection_probability.get(region, '0.00')}%"
        }
        table_data.append(row)

    table_data.append({
        "region": "total",
        "first_count": sum(int(v) for v in first_detection_count.values()),
        "second_count": sum(int(v) for v in second_detection_count.values()),
        "confidence": "100.00%"
    })

    return roi_analyze_result, table_data

def run_gradcam_roi_analysis(
    video_path,
    file_name,
    result,
    checkpoint_name='checkpoint_v35_best',
    selected_model='EfficientNet-b0',
    model=None,
    convert_videos=True,
):
    if model is None:
        model = load_model(selected_model, checkpoint_name, device)

    roi_analyze_result, table_data = calculate_roi_scores(video_path, file_name, result, model)

    if convert_videos:
        convert_gradcam_outputs(video_path)

    return roi_analyze_result, table_data
