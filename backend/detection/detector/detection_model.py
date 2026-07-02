# Model with feature visualization
import logging
import cv2
import torch
from torchvision import transforms
import os
import numpy as np
from .model_cache import get_cached_model


logger = logging.getLogger(__name__)

CPU_THREAD_LIMIT = int(os.environ.get("DEFAKE_CPU_THREADS", "4"))
INFERENCE_BATCH_SIZE = int(os.environ.get("DEFAKE_INFERENCE_BATCH_SIZE", "16"))

torch.set_num_threads(CPU_THREAD_LIMIT)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
cv2.setNumThreads(1)


def _predict_batch(input_tensors, model, device, model_lock):
    if not input_tensors:
        return [], [], [], []

    with torch.inference_mode():
        batch_tensor = torch.stack(input_tensors).unsqueeze(1).to(device).float()
        with model_lock:
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

def run_detection_model(video_path, selected_model='EfficientNet-b0', checkpoint_name='checkpoint_v35'):

    model, device, model_lock = get_cached_model(
        "inference",
        selected_model,
        checkpoint_name
    )
    logger.debug("Using device for inference: %s", device)

    if not os.path.exists(video_path):
        raise ValueError(f"비디오 파일이 존재하지 않습니다: {video_path}")

    video_name = os.path.basename(video_path)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    frame_probs = []
    frame_preds = []
    method_preds = []
    frame_scores = []

    cap = cv2.VideoCapture(video_path)
    batch = []

    success, frame = cap.read()
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch.append(transform(frame))

        if len(batch) >= INFERENCE_BATCH_SIZE:
            batch_probs, batch_preds, batch_methods, batch_scores = _predict_batch(batch, model, device, model_lock)
            frame_probs.extend(batch_probs)
            frame_preds.extend(batch_preds)
            method_preds.extend(batch_methods)
            frame_scores.extend(batch_scores)
            batch = []

        success, frame = cap.read()

    if batch:
        batch_probs, batch_preds, batch_methods, batch_scores = _predict_batch(batch, model, device, model_lock)
        frame_probs.extend(batch_probs)
        frame_preds.extend(batch_preds)
        method_preds.extend(batch_methods)
        frame_scores.extend(batch_scores)

    cap.release()

    method_dict = {0: 'original', 1: 'Deepfakes', 2: 'FaceShifter', 3: 'FaceSwap', 4: 'NeuralTextures', 5: 'Face2Face', 6: 'others'}

    # 두개의 예측
    final_prediction = 'Unknown' if len(frame_preds) == 0 else ('REAL' if round(sum(frame_preds)/len(frame_preds)) == 1 else 'FAKE')
    majority_method = max(set(method_preds), key=method_preds.count) if method_preds else 6
    
    logger.debug("Frame vote prediction before probability aggregation: %s", final_prediction)
    logger.debug("Majority method class before label mapping: %s", majority_method)

    # 비디오 하나에 대한 최종 예측
    frame_probs = np.array(frame_probs)
    if len(frame_preds) == 0:
        final_prediction = 'Unknown'
        final_probability = 0.0
        majority_method='Unknown'
    else:
        avg_probs = np.mean(frame_probs, axis=0)  # [mean_fake, mean_real]
        majority = round(sum(frame_preds) / len(frame_preds))  # 다수결
        logger.debug("Real-frame votes: %s/%s", sum(frame_preds), len(frame_preds))
        final_prediction = 'REAL' if majority == 1 else 'FAKE'
        final_probability = avg_probs[1] if final_prediction == 'REAL' else avg_probs[0]
        majority_method=method_dict[majority_method]


    result={
        'Filename': os.path.basename(video_path),
        'Filepath': video_path,
        'Prediction': final_prediction,
        'Probability': f"{final_probability * 100:.2f}",
        'Method': majority_method
    }

    return result
