import cv2
import torch
import torchvision.transforms as T
import numpy as np
import torch.nn.functional as F
from torchvision import models
from torch import nn
import os
import json
import glob
import pandas as pd
from tqdm import tqdm
from django.conf import settings
from pathlib import Path
import subprocess
import cv2
from concurrent.futures import ThreadPoolExecutor

from .model_cache import get_cached_model, get_device
from .preprocessing import ROI_METADATA_FILENAME

checkpoint_path=Path(__file__).resolve().parent

device = get_device()
print(f"Using device: {device}")

CPU_THREAD_LIMIT = int(os.environ.get("DEFAKE_CPU_THREADS", "4"))
GRADCAM_PIPELINE_BUFFER = int(os.environ.get("DEFAKE_GRADCAM_PIPELINE_BUFFER", "2"))

torch.set_num_threads(CPU_THREAD_LIMIT)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
cv2.setNumThreads(1)




# ✅ Grad-CAM computation for binary classification
def compute_gradcam_binary(model, input_tensor, model_lock=None, target_class=0, device_override=None):
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
    with model_lock or _null_context():
        _, binary_output, method_output = model(input_tensor)

        # Get the probability of the target class
        prob = F.softmax(binary_output, dim=1)[0, target_class].item()

        # Predict binary and method classes
        binary_pred = torch.argmax(binary_output, dim=1).item()   # 0: fake, 1: real
        method_pred = torch.argmax(method_output, dim=1).item()   # 0: original, 1~6: fake methods, 7: others

        # 🔴 Condition 1: If the prediction is real and the method is original, skip CAM computation / 조건 1: real(1) + original(0) → CAM X
        if binary_pred == 1 and method_pred==0:
            cam = np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]))
            f.remove()
            b.remove()
            return cam,prob,binary_pred, method_pred

        # Grad-CAM for fake class (target_class = 0)
        target_class = 0
        model.zero_grad()
        binary_output[0, target_class].backward()

    # Compute Grad-CAM
    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam = (weights * fmap).sum(dim=1, keepdim=True)
    cam = F.relu(cam).squeeze().cpu().numpy()

    # 🔵 Condition 2: If the prediction is fake and the method is not original, enhance CAM / 조건 2: fake (0) + method (1~7) (≠ 0) → CAM ↑
    if binary_pred == 0 and method_pred != 0:
        cam *=1.5

    # Normalize and resize the CAM
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))



    f.remove()
    b.remove()
    return cam, prob, binary_pred, method_pred


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False

def get_bbox(pts):
    x, y = pts[:,0], pts[:,1]
    return int(x.min()), int(y.min()), int(x.max()), int(y.max())

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

        print(f"✅ Video converted: {output_path}")
        if delete_source and os.path.exists(input_path):
            os.remove(input_path)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during conversion: {e}")


def _compute_gradcam_for_rgb(model, model_lock, transform, rgb):
    img = transform(rgb).to(device)
    return compute_gradcam_binary(model, img, model_lock=model_lock)


def _load_roi_metadata(video_dir):
    metadata_path = os.path.join(video_dir, ROI_METADATA_FILENAME)
    if not os.path.exists(metadata_path):
        print(f"ROI metadata not found: {metadata_path}")
        return []

    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    return metadata.get("frames", [])


def _default_roi_bboxes(width, height):
    return {
        "Jawline": [0, int(height * 0.58), width, height],
        "Left Eye": [int(width * 0.22), int(height * 0.30), int(width * 0.45), int(height * 0.42)],
        "Right Eye": [int(width * 0.55), int(height * 0.30), int(width * 0.78), int(height * 0.42)],
        "Left Eyebrow": [int(width * 0.20), int(height * 0.21), int(width * 0.45), int(height * 0.31)],
        "Right Eyebrow": [int(width * 0.55), int(height * 0.21), int(width * 0.80), int(height * 0.31)],
        "Nose": [int(width * 0.42), int(height * 0.40), int(width * 0.58), int(height * 0.63)],
        "Mouth": [int(width * 0.32), int(height * 0.66), int(width * 0.68), int(height * 0.80)],
    }


def _normalize_bbox_map(raw_bbox_map, width, height):
    if not raw_bbox_map:
        return _default_roi_bboxes(width, height)

    bbox_map = {}
    for region, bbox in raw_bbox_map.items():
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        bbox_map[region] = (x1, y1, x2, y2)

    for region, bbox in _default_roi_bboxes(width, height).items():
        bbox_map.setdefault(region, bbox)

    return bbox_map


def _write_processed_frame(
    pending_item,
    facial_region,
    first_detection_count,
    second_detection_count,
    detection_probabillity,
    video_writer_box,
    video_writer_cam,
):
    frame = pending_item["frame"]
    cam, cam_score, binary_pred, method_pred = pending_item["gradcam_future"].result()

    cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
    bbox_map = _normalize_bbox_map(
        pending_item.get("roi_bboxes"),
        frame.shape[1],
        frame.shape[0],
    )
    scores = {region: roi_activation(cam, box) for region, box in bbox_map.items()}
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    first_activated_region = sorted_scores[0][0]
    second_activated_region = sorted_scores[1][0]

    if scores[first_activated_region] <= 0.2:
        first_activated_region = "None"

    if scores[second_activated_region] <= 0.2:
        second_activated_region = "None"

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    frame_with_box = frame.copy()
    overlay_with_box = overlay.copy()

    if first_activated_region != "None":
        x1, y1, x2, y2 = bbox_map[first_activated_region]
        cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(overlay_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if second_activated_region != "None":
        x1, y1, x2, y2 = bbox_map[second_activated_region]
        cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.rectangle(overlay_with_box, (x1, y1), (x2, y2), (255, 0, 0), 2)

    video_writer_box.write(frame_with_box)
    video_writer_cam.write(overlay_with_box)

    first_detection_count[first_activated_region]+=1
    second_detection_count[second_activated_region]+=1
    for key in facial_region:
        if key!="None" and scores[key]!=-1:
            detection_probabillity[key]+=cam_score * scores[key]

    return True


def calculate_roi_scores(video_dir, file_name, result, model, model_lock):
    os.makedirs(video_dir, exist_ok=True)
    roi_metadata_frames = _load_roi_metadata(video_dir)

    transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    facial_region=['Jawline', 'Left Eye', 'Right Eye', 'Left Eyebrow', 'Right Eyebrow', 'Nose', 'Mouth','None']
    first_detection_count = {key: 0 for key in facial_region}
    second_detection_count = {key: 0 for key in facial_region}
    detection_probabillity={key: 0.0 for key in facial_region}

    video_path_file_name=os.path.join(video_dir,file_name)
    cap = cv2.VideoCapture(video_path_file_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_idx = 0
    processed_count = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_writer_box = cv2.VideoWriter(
        os.path.join(video_dir, "output_box_on_original.mp4"),
        fourcc, fps, (width, height)
    )

    video_writer_cam = cv2.VideoWriter(
        os.path.join(video_dir, "grad_cam_on_original.mp4"),
        fourcc, fps, (width, height)
    )

    pending = []

    with ThreadPoolExecutor(max_workers=1) as gradcam_executor:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            metadata_item = (
                roi_metadata_frames[frame_idx]
                if frame_idx < len(roi_metadata_frames)
                else {}
            )
            pending.append({
                "frame_idx": frame_idx,
                "frame": frame.copy(),
                "roi_bboxes": metadata_item.get("roi_bboxes"),
                "gradcam_future": gradcam_executor.submit(
                    _compute_gradcam_for_rgb,
                    model,
                    model_lock,
                    transform,
                    rgb
                ),
            })

            if len(pending) >= GRADCAM_PIPELINE_BUFFER:
                if _write_processed_frame(
                    pending.pop(0),
                    facial_region,
                    first_detection_count,
                    second_detection_count,
                    detection_probabillity,
                    video_writer_box,
                    video_writer_cam
                ):
                    processed_count += 1

            frame_idx += 1

        while pending:
            if _write_processed_frame(
                pending.pop(0),
                facial_region,
                first_detection_count,
                second_detection_count,
                detection_probabillity,
                video_writer_box,
                video_writer_cam
            ):
                processed_count += 1

    cap.release()
    video_writer_box.release()
    video_writer_cam.release()

    denominator = processed_count or 1
    first_detection_rate = {key: round((value / denominator)*100, 2) for key, value in first_detection_count.items()}
    second_detection_rate = {key: round((value / denominator)*100, 2) for key, value in second_detection_count.items()}
    raw_detection_probabillity= {key: round(value, 4) for key, value in detection_probabillity.items()}
    probabillity_total = sum(detection_probabillity.values())
    if probabillity_total:
        detection_probabillity= {key: round((value/probabillity_total)*100, 2) for key, value in detection_probabillity.items()}
    else:
        detection_probabillity= {key: 0.0 for key in detection_probabillity}

    roi_analyze_result = {
    "video_name": file_name,
    "binary_pred":result["Prediction"],
    "cam_score":result["Probability"],
    "method_pred":result["Method"],
    "facial_region": facial_region,
    "first_detection_count": first_detection_count,
    "second_detection_count": second_detection_count,
    "first_detection_rate": first_detection_rate,
    "second_detection_rate": second_detection_rate,
    "raw_detection_probability": raw_detection_probabillity,
    "detection_probability": detection_probabillity
    }

    table_data = []
    for region in facial_region:
        row = {
            "region": region,
            "first_count": f"{first_detection_count.get(region, '0')} ({first_detection_rate.get(region, '0.00')}%)",
            "second_count": f"{second_detection_count.get(region, '0')} ({second_detection_rate.get(region, '0.00')}%)",
            "confidence": "-" if detection_probabillity.get(region) is None else f"{detection_probabillity.get(region, '0.00')}%"
        }
        table_data.append(row)

    table_data.append({
        "region": "total",
        "first_count": sum(int(v) for v in first_detection_count.values()),
        "second_count": sum(int(v) for v in second_detection_count.values()),
        "confidence": "100.00%"
    })

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

    return roi_analyze_result, table_data

def analyze_roi_activation(video_dir, file_name, result, model):
    return calculate_roi_scores(video_dir, file_name, result, model, _null_context())



def run_gradcam_roi_analysis(video_path,file_name,result,checkpoint_name='checkpoint_v35',selected_model='EfficientNet-b0'):
    model, cached_device, model_lock = get_cached_model(
        "gradcam",
        selected_model,
        checkpoint_name
    )

    return calculate_roi_scores(video_path, file_name, result, model, model_lock)


