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

frames=100

checkpoint_path=Path(__file__).resolve().parent

class Model(nn.Module):
    def __init__(self, num_classes,model_name="resnext50_32x4d", lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        self.model_name = model_name 

        model = models.resnext50_32x4d(pretrained = True) #Residual Network CNN
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.latent_dim = 2048
        self.lstm = nn.LSTM(self.latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(hidden_dim,num_classes) # hidden_dim 변수로 넣어줌
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,self.latent_dim) # resnext50_32x4d, xception : 2048, efficientnet-b0 : 1280
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(torch.mean(x_lstm,dim = 1)))


def run_detection_model(video_path, selected_model='resnext50_32x4d', checkpoint_name='checkpoint_1'):
    video_name = os.path.basename(video_path)

    if torch.backends.mps.is_available():
        print("MPS is available. Using MPS.")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("CUDA is available. Using CUDA.")
        device = torch.device("cuda")
    else:
        print("CUDA and MPS not available. Using CPU.")
        device = torch.device("cpu")

    print(f"✅ Using device: {device}")


    # 모델 구조를 다시 정의
    model = Model(num_classes=2, model_name=selected_model).to(device)
    model.load_state_dict(torch.load(f'{checkpoint_path}/{checkpoint_name}.pt'))
    model.to(device)
    # 3. 평가 모드 전환
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])


    # 결과 저장 리스트
    results = []
    label_list = []
    folder_path_list=[]
    frame_probs = []

    with torch.no_grad():
        cap = cv2.VideoCapture(video_path)
        frame_preds = []
        frame_idx = 0

        success, frame = cap.read()

        while success:
            frame_idx += 1
            if frame_idx % 5 == 0:  # 매 5번째 프레임만 뽑아서 예측 (속도 + 대표성)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = transform(frame)
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # (batch=1, seq_len=1, c=3, h, w)
                input_tensor = input_tensor.to(device).float()

                fmap, outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                frame_probs.append(probs[0].cpu().numpy())  # [fake_prob, real_prob]
                _, predicted = torch.max(outputs, 1)


                frame_preds.append(predicted.item())

            success, frame = cap.read()

        cap.release()

        # 비디오 하나에 대한 최종 예측
        frame_probs = np.array(frame_probs)
        if len(frame_preds) == 0:
            final_prediction = 'Unknown'
            final_probability = 0.0
        else:
            avg_probs = np.mean(frame_probs, axis=0)  # [mean_fake, mean_real]
            majority = round(sum(frame_preds) / len(frame_preds))  # 다수결
            print(sum(frame_preds),"here!!!!!",len(frame_preds))
            final_prediction = 'REAL' if majority == 1 else 'FAKE'
            final_probability = avg_probs[1] if final_prediction == 'REAL' else avg_probs[0]

        result={
            'Filename': os.path.basename(video_path),
            'Filepath': video_path,
            'Prediction': final_prediction,
            'Probability': f"{final_probability * 100:.2f}"
        }

    return result
