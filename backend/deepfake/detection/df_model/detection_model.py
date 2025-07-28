#Model with feature visualization
from pathlib import Path
from django.conf import settings
import sys
import cv2
import torch
from torch import nn
from torchvision import models
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
import random
import numpy as np
from .model import Model

checkpoint_path=Path(__file__).resolve().parent

def run_detection_model(video_path, selected_model='EfficientNet-b0', checkpoint_name='checkpoint_v35'):

    # ✅ Set the device to MPS(for Mac) if available, otherwise fallback to CUDA or CPU
    device = torch.device("mps") if torch.backends.mps.is_available() else (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}")


    # 모델 구조를 정의
    model = Model(num_binary_classes=2, num_method_classes=7, model_name=selected_model).to(device)
    model.load_state_dict(torch.load(f'{checkpoint_path}/{checkpoint_name}.pt', map_location=device))
    model.eval()

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


    with torch.no_grad():
        cap = cv2.VideoCapture(video_path)
        frame_preds = []
        method_preds=[]
        pooled_features_per_video = []
        frame_scores = []
        frame_idx = 0

        success, frame = cap.read()

        while success:
            frame_idx += 1
            if frame_idx % 1 == 0:  # 매 5번째 프레임만 뽑아서 예측 (속도 + 대표성)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = transform(frame)
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # (batch=1, seq_len=1, c=3, h, w)
                input_tensor = input_tensor.to(device).float()
                fmap, output_bin, output_method = model(input_tensor)

                probs = torch.softmax(output_bin, dim=1)
                frame_probs.append(probs[0].cpu().numpy())  # [fake_prob, real_prob]

                _, predicted_bin = torch.max(output_bin, 1)
                _, predicted_method = torch.max(output_method, 1)

                # 추가: threshold 기반 unknown 분류 처리
                method_probs = torch.softmax(output_method.squeeze(0), dim=0)
                method_confidence, method_class = torch.max(method_probs, dim=0)
                threshold = 0.5  # ← 원하는 값으로 조절

                if method_confidence < threshold:
                    predicted_method = torch.tensor([6])  # unknown class
                else:
                    predicted_method = method_class.unsqueeze(0)  # 그대로 유지


                score = torch.softmax(output_bin.squeeze(0), dim=0)[1].item()  # Real 확률만
                frame_scores.append(score)

                frame_preds.append(predicted_bin.item())
                method_preds.append(predicted_method.item())

            success, frame = cap.read()

        cap.release()

        method_dict = {0: 'original', 1: 'Deepfakes', 2: 'FaceShifter', 3: 'FaceSwap', 4: 'NeuralTextures', 5: 'Face2Face', 6: 'others'}

        # 두개의 예측
        final_prediction = 'Unknown' if len(frame_preds) == 0 else ('REAL' if round(sum(frame_preds)/len(frame_preds)) == 1 else 'FAKE')
        majority_method = max(set(method_preds), key=method_preds.count) if method_preds else 6
        
        print("Final Prediction : ",final_prediction)
        print("Method Predictin : ",majority_method)

        # 비디오 하나에 대한 최종 예측
        frame_probs = np.array(frame_probs)
        if len(frame_preds) == 0:
            final_prediction = 'Unknown'
            final_probability = 0.0
            majority_method='Unknown'
        else:
            avg_probs = np.mean(frame_probs, axis=0)  # [mean_fake, mean_real]
            majority = round(sum(frame_preds) / len(frame_preds))  # 다수결
            print(sum(frame_preds),"here!!!!!",len(frame_preds)) # 진짜라고 판단한 개수
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
