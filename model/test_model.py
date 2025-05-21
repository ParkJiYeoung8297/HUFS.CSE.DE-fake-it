# # 원격 서버
# import sys
# selected_model = sys.argv[1]
# use_input1= int(sys.argv[2])  # 첫 번째 인자 [0,1] / 1이면 사용, 0이면 사용 X
# use_input2= int(sys.argv[3])  # 두 번째 인자 [0,1]
# test_input_file_path='/root/jiyeong/Dataset/ff++/val/*'
# test_input_file_path2='/root/jiyeong/Dataset/DFDC/val/*'
# checkpoint_path='/root/jiyeong/model/checkpoints'
# checkpoint_name=sys.argv[4]
# base_path = '/root/jiyeong/Dataset'  # 상대 주소 찾기 위해 base_path 제거
# frames=100

# 로컬
import sys
selected_model = sys.argv[1]
use_input1= int(sys.argv[2])  # 첫 번째 인자 [0,1] / 1이면 사용, 0이면 사용 X
use_input2= int(sys.argv[3])  # 두 번째 인자 [0,1]
test_input_file_path=f'/Users/jiyeong/Desktop/컴공 캡스톤/Dataset/ff++/val/*'
test_input_file_path2=f'/Users/jiyeong/Desktop/컴공 캡스톤/Dataset/DFDC/val/*'
checkpoint_path=f'/Users/jiyeong/HUFS.CSE.DE-fake-it/model/checkpoints'
checkpoint_name=sys.argv[4]
base_path = '/root/jiyeong/Dataset'  # 상대 주소 찾기 위해 base_path 제거
frames=100


print("Check parameter")
print(f"ff++: {'use' if use_input1 == 1 else 'not use'}")
print(f"dfdc: {'use' if use_input2 == 1 else 'not use'}")
print(f"Checkpoint name: {checkpoint_name}")
print()

data_list=[]


#Model with feature visualization
import cv2
import torch
from torch import nn
from torchvision import models
from torchvision import transforms
import matplotlib.pyplot as plt

import timm
from efficientnet_pytorch import EfficientNet

class Model(nn.Module):
    def __init__(self, num_classes,model_name="resnext50_32x4d", lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        self.model_name = model_name 

        if self.model_name=="resnext50_32x4d":
          model = models.resnext50_32x4d(pretrained = True) #Residual Network CNN
          self.model = nn.Sequential(*list(model.children())[:-2])
          self.latent_dim = 2048
        elif self.model_name=="xception":
          self.latent_dim = 2048 # xception
          model = timm.create_model('xception', pretrained=True, features_only=False)
          self.model = nn.Sequential(*list(model.children())[:-2])  # or model.forward_features
        elif self.model_name=="EfficientNet-b0":
           self.latent_dim = 1280 # efficient
           model = EfficientNet.from_pretrained('efficientnet-b0')
           self.model = model.extract_features
        print("latet_dim: ",self.latent_dim)

           

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
    
def get_device():
    if torch.backends.mps.is_available():
        print("MPS is available. Using MPS.")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("CUDA is available. Using CUDA.")
        device = torch.device("cuda")
    else:
        print("CUDA and MPS not available. Using CPU.")
        device = torch.device("cpu")
    return device

# 디바이스 설정
device = get_device()
print(f"✅ Using device: {device}")


# 모델 구조를 다시 정의
model = Model(num_classes=2, model_name=selected_model).to(device)
# checkpoint 불러오기
model.load_state_dict(torch.load(f'{checkpoint_path}/{checkpoint_name}.pt'))
# 3. 평가 모드 전환
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

#Output confusion matrix   성능 평가
import seaborn as sn
from sklearn.metrics import confusion_matrix  #내가 추가함
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print('\n')
    df_cm = pd.DataFrame(cm, range(2), range(2))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True,fmt='d', annot_kws={"size": 16}) # font size ,fmt='d'로 정수 표현
    plt.ylabel('Actual label', size = 20)
    plt.xlabel('Predicted label', size = 20)
    plt.xticks(np.arange(2), ['Fake', 'Real'], size = 16)
    plt.yticks(np.arange(2), ['Fake', 'Real'], size = 16)
    plt.ylim([2, 0])
    # plt.show()
    plt.savefig(f'{checkpoint_path}/(test)_{checkpoint_name}_plot.png')

    calculated_acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+ cm[1][1])
    print("Calculated Accuracy",calculated_acc*100)


    y_true = (['Fake'] * sum(cm[0]) + ['Real'] * sum(cm[1]))
    y_pred = (['Fake'] * cm[0][0] + ['Real'] * cm[0][1] +
            ['Fake'] * cm[1][0] + ['Real'] * cm[1][1])

    # 성능 출력
    print("📊 Confusion Matrix:\n", cm)
    print("\n📈 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Fake', 'Real']))


import torch
from torchvision import transforms
import cv2
import os
import pandas as pd
import glob
import random

if use_input1==1 and use_input2==1:
    new_video_files =  glob.glob(f'{test_input_file_path}/*.mp4')   # 경로 변경
    data_list.append(test_input_file_path)
    new_video_files += glob.glob(f'{test_input_file_path2}/*.mp4') 
    data_list.append(test_input_file_path2)
elif use_input1==1:
    new_video_files =  glob.glob(f'{test_input_file_path}/*.mp4')   # 경로 변경
    data_list.append(test_input_file_path)
elif use_input2==1:
  new_video_files = glob.glob(f'{test_input_file_path2}/*.mp4') 
  data_list.append(test_input_file_path2)
# new_video_files += glob.glob(f'{test_output_file_path}/*.mp4')
# video_files += glob.glob('/content/drive/My Drive/DFDC_FAKE_Face_only_data/*.mp4')
# video_files += glob.glob('/content/drive/My Drive/DFDC_REAL_Face_only_data/*.mp4')
random.shuffle(new_video_files)
random.shuffle(new_video_files)

frame_count = []
short_frame=[]

for video_file in reversed(new_video_files): # 이거 앞에서 부터 하면 remove로 인해 frame_count랑 video_files 길이가 달라짐, 그래서 reversed 추가하여 뒤에서 부터 탐색!!
  cap = cv2.VideoCapture(video_file)
  if(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))<frames):  # frames 변수 위에서 조정
    new_video_files.remove(video_file)
    short_frame.append(video_file)
    continue

  frame_count.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
  
# print("frames are " , frame_count)
print("Total no of video: " , len(frame_count))
print('Average frame per video:',np.mean(frame_count))
print('Short_frame_count : ', len(short_frame))

from tqdm import tqdm
# 결과 저장 리스트
results = []
label_list = []
folder_path_list=[]

with torch.no_grad():
    for video_path in tqdm(new_video_files):
        cap = cv2.VideoCapture(video_path)
        frame_preds = []
        frame_idx = 0

        relative_path = os.path.relpath(video_path,base_path).replace("\\", "/")
        folder_path_list.append(relative_path)

        # label (real/fake)
        if 'real' in relative_path.lower():
            label = 'REAL'
        elif 'fake' in relative_path.lower():
            label = 'FAKE'
        else:
            label = 'unknown'
        label_list.append(label)
        success, frame = cap.read()
        while success:
            frame_idx += 1
            if frame_idx % 5 == 0:  # 매 5번째 프레임만 뽑아서 예측 (속도 + 대표성)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = transform(frame)
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # (batch=1, seq_len=1, c=3, h, w)
                input_tensor = input_tensor.to(device).float()

                fmap, outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)

                frame_preds.append(predicted.item())

            success, frame = cap.read()

        cap.release()

        # 비디오 하나에 대한 최종 예측
        if len(frame_preds) == 0:
            final_prediction = 'Unknown'
        else:
            majority = round(sum(frame_preds) / len(frame_preds))  # 다수결
            final_prediction = 'REAL' if majority == 1 else 'FAKE'

        results.append({
            'Filename': os.path.basename(video_path),
            'Filepath': video_path,
            'label': label,
            'Prediction': final_prediction
        })

# 결과 DataFrame으로 만들기
df = pd.DataFrame(results)

# 엑셀로 저장
output_excel_path = f'{checkpoint_path}/(test)_{checkpoint_name}_predictions.xlsx'
df.to_excel(output_excel_path, index=False, engine='openpyxl')

print(f"✅ 모든 비디오 예측 결과가 엑셀로 저장되었습니다: {output_excel_path}")

# Confusion Matrix 계산
labels = ['REAL', 'FAKE']
y_true = label_list  # 실제 레이블
y_pred = [result['Prediction'] for result in results]  # 예측 레이블


print_confusion_matrix(y_true,y_pred)
