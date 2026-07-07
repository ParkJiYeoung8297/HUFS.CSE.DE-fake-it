import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from facenet_pytorch import MTCNN
from torchvision import transforms

from .modeling import ResearchModel, get_device


ROI_METADATA_FILENAME = "roi_metadata.json"
FRAME_SAMPLE_STRIDE = max(1, int(os.environ.get("DEFAKE_FRAME_SAMPLE_STRIDE", "5")))
INFERENCE_BATCH_SIZE = int(os.environ.get("DEFAKE_INFERENCE_BATCH_SIZE", "16"))
GRADCAM_PIPELINE_BUFFER = int(os.environ.get("DEFAKE_GRADCAM_PIPELINE_BUFFER", "2"))


def load_pairwise_model(checkpoint_path, model_name="EfficientNet-b0"):
    device = get_device()
    model = ResearchModel(num_binary_classes=2, num_method_classes=7, model_name=model_name).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, device


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


def _default_roi_bboxes(width=224, height=224):
    return {
        "Jawline": [0, int(height * 0.58), width, height],
        "Left Eye": [int(width * 0.22), int(height * 0.30), int(width * 0.45), int(height * 0.42)],
        "Right Eye": [int(width * 0.55), int(height * 0.30), int(width * 0.78), int(height * 0.42)],
        "Left Eyebrow": [int(width * 0.20), int(height * 0.21), int(width * 0.45), int(height * 0.31)],
        "Right Eyebrow": [int(width * 0.55), int(height * 0.21), int(width * 0.80), int(height * 0.31)],
        "Nose": [int(width * 0.42), int(height * 0.40), int(width * 0.58), int(height * 0.63)],
        "Mouth": [int(width * 0.32), int(height * 0.66), int(width * 0.68), int(height * 0.80)],
    }


def _build_roi_bboxes_from_mtcnn(face_box, landmarks, output_size=224):
    x1, y1, x2, y2 = [float(value) for value in face_box]
    face_w = max(x2 - x1, 1.0)
    face_h = max(y2 - y1, 1.0)

    def scale_point(point):
        px, py = point
        return (
            _clamp((float(px) - x1) / face_w * output_size, 0, output_size),
            _clamp((float(py) - y1) / face_h * output_size, 0, output_size),
        )

    left_eye, right_eye, nose, mouth_left, mouth_right = [scale_point(point) for point in landmarks]
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
    mtcnn = MTCNN(keep_all=False, device=torch.device("cpu"))
    output_fps = max(1, round(30 / FRAME_SAMPLE_STRIDE))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"MJPG"), output_fps, (224, 224))
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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _, landmarks = mtcnn.detect(rgb, landmarks=True)
            if boxes is not None:
                x1, y1, x2, y2 = map(int, boxes[0])
                h, w, _ = frame.shape
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                face = frame[y1:y2, x1:x2]
                if face.size != 0:
                    resized = cv2.resize(face, (224, 224))
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
        finally:
            source_frame_idx += 1

    out.release()
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False)
    return output_video_path


def _predict_batch(input_tensors, model, device):
    if not input_tensors:
        return [], [], [], []
    with torch.inference_mode():
        batch_tensor = torch.stack(input_tensors).unsqueeze(1).to(device).float()
        _, output_bin, output_method = model(batch_tensor)
        probs = torch.softmax(output_bin, dim=1)
        method_probs = torch.softmax(output_method, dim=1)
        method_confidences, method_classes = torch.max(method_probs, dim=1)
        _, predicted_bins = torch.max(output_bin, 1)
    frame_probs = probs.cpu().numpy().tolist()
    frame_preds = predicted_bins.cpu().numpy().tolist()
    frame_scores = probs[:, 1].cpu().numpy().tolist()
    method_preds = [
        6 if confidence.item() < 0.5 else method_class.item()
        for confidence, method_class in zip(method_confidences, method_classes)
    ]
    return frame_probs, frame_preds, method_preds, frame_scores


def run_detection_model(video_path, model, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    frame_probs = []
    frame_preds = []
    method_preds = []
    cap = cv2.VideoCapture(video_path)
    batch = []

    success, frame = cap.read()
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch.append(transform(frame))
        if len(batch) >= INFERENCE_BATCH_SIZE:
            batch_probs, batch_preds, batch_methods, _ = _predict_batch(batch, model, device)
            frame_probs.extend(batch_probs)
            frame_preds.extend(batch_preds)
            method_preds.extend(batch_methods)
            batch = []
        success, frame = cap.read()

    if batch:
        batch_probs, batch_preds, batch_methods, _ = _predict_batch(batch, model, device)
        frame_probs.extend(batch_probs)
        frame_preds.extend(batch_preds)
        method_preds.extend(batch_methods)

    cap.release()
    method_dict = {0: "original", 1: "Deepfakes", 2: "FaceShifter", 3: "FaceSwap", 4: "NeuralTextures", 5: "Face2Face", 6: "others"}
    frame_probs = np.array(frame_probs)
    if len(frame_preds) == 0:
        final_prediction = "Unknown"
        final_probability = 0.0
        majority_method = "Unknown"
    else:
        avg_probs = np.mean(frame_probs, axis=0)
        majority = round(sum(frame_preds) / len(frame_preds))
        final_prediction = "REAL" if majority == 1 else "FAKE"
        final_probability = avg_probs[1] if final_prediction == "REAL" else avg_probs[0]
        majority_method = max(set(method_preds), key=method_preds.count) if method_preds else 6
        majority_method = method_dict[majority_method]
    return {
        "Filename": os.path.basename(video_path),
        "Filepath": video_path,
        "Prediction": final_prediction,
        "Probability": f"{final_probability * 100:.2f}",
        "Method": majority_method,
    }


def compute_gradcam_binary(model, input_tensor, device, target_class=0):
    fmap = None
    grad = None

    def fw_hook(module, inp, out):
        nonlocal fmap
        fmap = out.detach()

    def bw_hook(module, grad_in, grad_out):
        nonlocal grad
        grad = grad_out[0].detach()

    last_layer = model.model[-1]
    forward_hook = last_layer.register_forward_hook(fw_hook)
    backward_hook = last_layer.register_backward_hook(bw_hook)
    input_tensor = input_tensor.to(device).unsqueeze(0).unsqueeze(0).requires_grad_(True)
    _, binary_output, method_output = model(input_tensor)
    prob = F.softmax(binary_output, dim=1)[0, target_class].item()
    binary_pred = torch.argmax(binary_output, dim=1).item()
    method_pred = torch.argmax(method_output, dim=1).item()

    if binary_pred == 1 and method_pred == 0:
        cam = np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]))
        forward_hook.remove()
        backward_hook.remove()
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
    forward_hook.remove()
    backward_hook.remove()
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


def _convert_video(input_path, output_path, delete_source=False):
    try:
        subprocess.run(["ffmpeg", "-i", input_path, "-vcodec", "libx264", "-acodec", "aac", "-strict", "experimental", "-y", output_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if delete_source and os.path.exists(input_path):
            os.remove(input_path)
    except subprocess.CalledProcessError:
        pass


def _write_processed_frame(pending_item, facial_region, first_detection_count, second_detection_count, detection_probability, video_writer_box, video_writer_cam):
    frame = pending_item["frame"]
    cam, cam_score, binary_pred, method_pred = pending_item["gradcam_future"].result()
    cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
    bbox_map = _normalize_bbox_map(pending_item.get("roi_bboxes"), frame.shape[1], frame.shape[0])
    scores = {region: roi_activation(cam, box) for region, box in bbox_map.items()}
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
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
    first_detection_count[first_activated_region] += 1
    second_detection_count[second_activated_region] += 1
    for key in facial_region:
        if key != "None" and scores[key] != -1:
            detection_probability[key] += cam_score * scores[key]
    return True


def run_gradcam_roi_analysis(video_dir, file_name, result, model, device):
    metadata_path = os.path.join(video_dir, ROI_METADATA_FILENAME)
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            roi_metadata_frames = json.load(metadata_file).get("frames", [])
    else:
        roi_metadata_frames = []
    transform = T.Compose([T.ToTensor(), T.Resize((224, 224)), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    facial_region = ["Jawline", "Left Eye", "Right Eye", "Left Eyebrow", "Right Eyebrow", "Nose", "Mouth", "None"]
    first_detection_count = {key: 0 for key in facial_region}
    second_detection_count = {key: 0 for key in facial_region}
    detection_probability = {key: 0.0 for key in facial_region}
    cap = cv2.VideoCapture(os.path.join(video_dir, file_name))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer_box = cv2.VideoWriter(os.path.join(video_dir, "output_box_on_original.mp4"), fourcc, fps, (width, height))
    video_writer_cam = cv2.VideoWriter(os.path.join(video_dir, "grad_cam_on_original.mp4"), fourcc, fps, (width, height))
    pending = []
    frame_idx = 0
    processed_count = 0

    with ThreadPoolExecutor(max_workers=1) as gradcam_executor:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            metadata_item = roi_metadata_frames[frame_idx] if frame_idx < len(roi_metadata_frames) else {}
            pending.append({
                "frame": frame.copy(),
                "roi_bboxes": metadata_item.get("roi_bboxes"),
                "gradcam_future": gradcam_executor.submit(lambda image=rgb: compute_gradcam_binary(model, transform(image).to(device), device)),
            })
            if len(pending) >= GRADCAM_PIPELINE_BUFFER:
                if _write_processed_frame(pending.pop(0), facial_region, first_detection_count, second_detection_count, detection_probability, video_writer_box, video_writer_cam):
                    processed_count += 1
            frame_idx += 1
        while pending:
            if _write_processed_frame(pending.pop(0), facial_region, first_detection_count, second_detection_count, detection_probability, video_writer_box, video_writer_cam):
                processed_count += 1

    cap.release()
    video_writer_box.release()
    video_writer_cam.release()
    denominator = processed_count or 1
    first_detection_rate = {key: round((value / denominator) * 100, 2) for key, value in first_detection_count.items()}
    second_detection_rate = {key: round((value / denominator) * 100, 2) for key, value in second_detection_count.items()}
    raw_detection_probability = {key: round(value, 4) for key, value in detection_probability.items()}
    probability_total = sum(detection_probability.values())
    detection_probability = {key: round((value / probability_total) * 100, 2) for key, value in detection_probability.items()} if probability_total else {key: 0.0 for key in detection_probability}
    roi_analyze_result = {
        "video_name": file_name,
        "binary_pred": result["Prediction"],
        "cam_score": result["Probability"],
        "method_pred": result["Method"],
        "facial_region": facial_region,
        "first_detection_count": first_detection_count,
        "second_detection_count": second_detection_count,
        "first_detection_rate": first_detection_rate,
        "second_detection_rate": second_detection_rate,
        "raw_detection_probability": raw_detection_probability,
        "detection_probability": detection_probability,
    }
    table_data = []
    for region in facial_region:
        table_data.append({
            "region": region,
            "first_count": f"{first_detection_count.get(region, '0')} ({first_detection_rate.get(region, '0.00')}%)",
            "second_count": f"{second_detection_count.get(region, '0')} ({second_detection_rate.get(region, '0.00')}%)",
            "confidence": "-" if detection_probability.get(region) is None else f"{detection_probability.get(region, '0.00')}%",
        })
    table_data.append({"region": "total", "first_count": sum(int(value) for value in first_detection_count.values()), "second_count": sum(int(value) for value in second_detection_count.values()), "confidence": "100.00%"})
    _convert_video(os.path.join(video_dir, "grad_cam_on_original.mp4"), os.path.join(video_dir, "converted_grad_cam_on_original.mp4"), delete_source=True)
    _convert_video(os.path.join(video_dir, "output_box_on_original.mp4"), os.path.join(video_dir, "converted_output_box_on_original.mp4"), delete_source=True)
    return roi_analyze_result, table_data
