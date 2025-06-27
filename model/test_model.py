# # 코랩 서버
# import sys
# selected_model = sys.argv[1]
# use_input1= int(sys.argv[2])  # 첫 번째 인자 [0,1] / 1이면 사용, 0이면 사용 X
# use_input2= int(sys.argv[3])  # 두 번째 인자 [0,1]
# test_input_file_path='/content/drive/MyDrive/Capstone/Dataset/ff++/test/*/*'
# test_input_file_path2='/content/drive/MyDrive/Capstone/Dataset/DFDC/test/*/*'
# checkpoint_path='/content/drive/MyDrive/Capstone/checkpoints'
# checkpoint_name=sys.argv[4]
# meta_data_path='/content/drive/MyDrive/Capstone/Dataset/ff++'
# base_path = '/content/drive/MyDrive/Capstone/Dataset'  # 상대 주소 찾기 위해 base_path 제거
# frames=150

# # 로컬
import sys
selected_model = "resnext50_32x4d"
test_input_file_path=f'/Users/jiyeong/Desktop/컴공 캡스톤/Dataset/ff++/test/*/*'
# test_input_file_path2=f'/Users/jiyeong/Desktop/컴공 캡스톤/Dataset/DFDC/val/*'
checkpoint_path=f'/Users/jiyeong/HUFS.CSE.DE-fake-it/model/checkpoints'
checkpoint_name="checkpoint_v3"
meta_data_path=f'/Users/jiyeong/Desktop/컴공 캡스톤/Dataset'
frames=150
base_path='/Users/jiyeong/Desktop/컴공 캡스톤/Dataset'


# # sys.stdout.reconfigure(line_buffering=True)  # 모든 print문에 flush=true 설정 반영
# # # 코랩 서버
# import sys
# selected_model = "resnext50_32x4d"
# test_input_file_path='/content/drive/MyDrive/Capstone/Dataset/ff++/test/*/*'
# # test_input_file_path2='/content/drive/MyDrive/Capstone/Dataset/DFDC/test/*/*'
# # test_input_file_path3='/content/drive/MyDrive/Capstone/Dataset/celeb-df/test/*/*'
# checkpoint_path='/content/drive/MyDrive/Capstone/checkpoints'
# checkpoint_name="checkpoint_v3"
# meta_data_path='/content/drive/MyDrive/Capstone/Dataset/ff++'
# base_path = '/content/drive/MyDrive/Capstone/Dataset'  # 상대 주소 찾기 위해 base_path 제거
# frames=150

print("Check parameter")
print(f"Dataset: FaceForencis++")
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
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import cv2
import os
import pandas as pd
import glob
import random
from tqdm import tqdm
import timm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
# from efficientnet_pytorch import EfficientNet
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class Model(nn.Module):
    def __init__(self, num_binary_classes=2, num_method_classes=7,model_name="resnext50_32x4d", lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
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
           #  model = EfficientNet.from_pretrained('efficientnet-b0')
           #  self.model = model.extract_features
           weights = EfficientNet_B0_Weights.DEFAULT
           model = efficientnet_b0(weights=weights)
           self.model = nn.Sequential(*list(model.features))
        print("latet_dim: ",self.latent_dim)
        self.lstm = nn.LSTM(self.latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()    # 이거는 넣고 빼고 실험해보래
        self.dp = nn.Dropout(0.4)
        # self.linear1 = nn.Linear(hidden_dim,num_classes) # hidden_dim 변수로 넣어줌
        self.avgpool = nn.AdaptiveAvgPool2d(1)


        # 두 개의 출력: 이진 분류와 method 분류
        self.binary_classifier = nn.Linear(hidden_dim, num_binary_classes)
        self.method_classifier = nn.Linear(hidden_dim, num_method_classes)

    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,self.latent_dim) # resnext50_32x4d, xception : 2048, efficientnet-b0 : 1280
        x_lstm,_ = self.lstm(x,None)
        # return fmap,self.dp(self.linear1(torch.mean(x_lstm,dim = 1)))
        pooled = torch.mean(x_lstm, dim=1)
        return fmap, self.binary_classifier(self.dp(pooled)), self.method_classifier(self.dp(pooled))

    
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
model = Model(num_binary_classes=2, num_method_classes=7, model_name=selected_model).to(device)
# checkpoint 불러오기
model.load_state_dict(torch.load(f'{checkpoint_path}/{checkpoint_name}.pt'))
# 3. 평가 모드 전환
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

#Output confusion matrix   성능 평가
import seaborn as sn
from sklearn.metrics import confusion_matrix  #내가 추가함
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

#Output confusion matrix / 모델 성능 평가

def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print('\n')
    df_cm = pd.DataFrame(cm, range(2), range(2))
    plt.clf()  # Clear the previous figure
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True,fmt='d', annot_kws={"size": 16}) # font size ,fmt='d'로 정수 표현
    plt.ylabel('Actual label', size = 20)
    plt.xlabel('Predicted label', size = 20)
    plt.xticks(np.arange(2), ['Fake', 'Real'], size = 16)
    plt.yticks(np.arange(2), ['Fake', 'Real'], size = 16)
    plt.ylim([2, 0])
    # plt.show()
    plt.savefig(f'{checkpoint_path}/{checkpoint_name}_plot(test).png')
    calculated_acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+ cm[1][1])
    print("Calculated Accuracy",calculated_acc*100)


    y_true = (['Fake'] * sum(cm[0]) + ['Real'] * sum(cm[1]))
    y_pred = (['Fake'] * cm[0][0] + ['Real'] * cm[0][1] +
            ['Fake'] * cm[1][0] + ['Real'] * cm[1][1])

    # 성능 출력
    print("📊 Confusion Matrix:\n", cm)
    print("\n📈 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Fake', 'Real']))

def print_confusion_matrix_method(y_true_method, y_pred_method):
    
    labels = ['original', 'Deepfakes', 'FaceShifter', 'FaceSwap', 'NeuralTextures', 'Face2Face','unknown']
    label_indices = list(range(len(labels))) # [0, 1, 2, ..., 6]
    cm = confusion_matrix(y_true_method, y_pred_method, labels=label_indices)
    print('\n')
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.clf()  # Clear the previous figure
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, fmt='d', annot_kws={"size": 12}, cmap='Blues')
    plt.ylabel('Actual label', size=16)
    plt.xlabel('Predicted label', size=16)
    plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45, ha='right', fontsize=12)
    plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0, fontsize=12)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{checkpoint_path}/{checkpoint_name}_plot(method)(test).png')

    # 정확도 계산: 모든 정답 예측 수 / 전체 샘플 수
    correct_preds = np.trace(cm)
    total_preds = np.sum(cm)
    calculated_acc = correct_preds / total_preds
    print(f"\n✅ Calculated Accuracy: {calculated_acc * 100:.2f}%")

    # 성능 출력
    print("📊 Confusion Matrix:\n", cm)
    print("\n📈 Classification Report:")
    print(classification_report(y_true_method, y_pred_method, target_names=labels, labels=label_indices))


#

def plot_roc_curve(true_bin, output_bin, checkpoint_path, checkpoint_name):
    pred_score = output_bin.cpu().numpy()  # Real 확률
    fpr, tpr, _ = roc_curve(true_bin, pred_score)
    roc_auc = auc(fpr, tpr)

    plt.clf()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Binary Classification)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{checkpoint_path}/{checkpoint_name}_roc_curve(test).png")
    print(f"✅ ROC Curve saved to {checkpoint_path}/{checkpoint_name}_roc_curve(test).png")

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def plot_tsne(features, labels, checkpoint_path, checkpoint_name, method_labels=None):
    n_samples = features.shape[0]
    perplexity = min(30, max(5, n_samples // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    features = np.array(features)
    labels = np.array(labels)
    X_embedded = tsne.fit_transform(features)

    plt.clf()
    plt.figure(figsize=(8, 6))
    unique_classes = np.unique(labels)
    for cls in unique_classes:
        idx = labels == cls
        label_name = method_labels[cls] if method_labels and cls < len(method_labels) else str(cls)
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=label_name, alpha=0.7)

    plt.legend()
    plt.title('t-SNE of Method Class Features')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{checkpoint_path}/{checkpoint_name}_tsne(test).png")
    print(f"✅ t-SNE plot saved to {checkpoint_path}/{checkpoint_name}_tsne(test).png")

#2. to load preprocessod video to memory / 전처리된 영상 가져오기
new_video_files =  glob.glob(f'{test_input_file_path}/*.mp4')

random.shuffle(new_video_files)

# ✅ 결과 저장 리스트 초기화  
method_pred_list = []  # ROC Curve 용
video_bin_scores = []   # t-SNE 시각화용

# ✅ 결과 저장 리스트 초기화
results = []
label_list = []
folder_path_list = []
method_list = []


video_feature_array = []
with torch.no_grad():
    for video_path in tqdm(new_video_files):
        cap = cv2.VideoCapture(video_path)
        frame_preds = []
        method_preds=[]
        pooled_features_per_video = []
        frame_scores = []

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

        # method (original/Deepfakes/FaceShifter/FaceSwap/NeuralTextures/Face2Face/unknown)
        if 'original' in relative_path.lower():
            method = 'original'
        elif 'deepfakes' in relative_path.lower():
            method = 'Deepfakes'
        elif 'faceshifter' in relative_path.lower():
            method = 'FaceShifter'
        elif 'faceswap' in relative_path.lower():
            method = 'FaceSwap'
        elif 'neuraltextures' in relative_path.lower():
            method = 'NeuralTextures'
        elif 'face2face' in relative_path.lower():
            method = 'Face2Face'
        else:
            method = 'unknown'
        method_list.append(method)

        success, frame = cap.read()


        while success:
            frame_idx += 1
            if frame_idx % 5 == 0:  # 매 5번째 프레임만 뽑아서 예측 (속도 + 대표성)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = transform(frame)
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # (batch=1, seq_len=1, c=3, h, w)
                input_tensor = input_tensor.to(device).float()

                fmap, output_bin, output_method = model(input_tensor)
                _, predicted_bin = torch.max(output_bin, 1)
                _, predicted_method = torch.max(output_method, 1)

                score = torch.softmax(output_bin.squeeze(0), dim=0)[1].item()  # Real 확률만
                frame_scores.append(score)

                frame_preds.append(predicted_bin.item())
                method_preds.append(predicted_method.item())

                # ✅ feature 및 확률 저장 (추가된 부분)
                # output_bin_all.append(output_bin.squeeze(0).detach().cpu())
                # pooled_feature = torch.mean(fmap.view(fmap.size(0), fmap.size(1), -1), dim=2)
                # feature_array.append(pooled_feature.squeeze(0).detach().cpu().numpy())
                pooled = torch.mean(fmap.view(fmap.size(0), fmap.size(1), -1), dim=2)
                pooled_features_per_video.append(pooled.squeeze(0).detach().cpu().numpy())

            success, frame = cap.read()
        
        # ⬇️ 프레임 평균을 비디오 feature로 저장
        if pooled_features_per_video:
            avg_feature = np.mean(pooled_features_per_video, axis=0)
            video_feature_array.append(avg_feature)
        
        if frame_scores:
            video_bin_scores.append(np.mean(frame_scores))

        cap.release()
        final_prediction = 'Unknown' if len(frame_preds) == 0 else ('REAL' if round(sum(frame_preds)/len(frame_preds)) == 1 else 'FAKE')
        majority_method = max(set(method_preds), key=method_preds.count) if method_preds else 6
        method_pred_list.append(majority_method)

        results.append({
            'Filename': os.path.basename(video_path),
            'Filepath': video_path,
            'label': label,
            'Prediction': final_prediction,
            'method': method,  # 실제 method
            'Predicted_method': majority_method  # 예측된 method
        })

# 결과 엑셀로 저장
output_excel_path = f'{checkpoint_path}/(test)_{checkpoint_name}_predictions.xlsx'
df = pd.DataFrame(results)
df.to_excel(output_excel_path, index=False, engine='openpyxl')

print(f"✅ 모든 비디오 예측 결과가 엑셀로 저장되었습니다: {output_excel_path}")

y_true = label_list
y_pred = [r['Prediction'] for r in results]
true_bin = [0 if l == 'FAKE' else 1 for l in y_true]
pred_bin = [0 if p == 'FAKE' else 1 for p in y_pred]
method_dict = {'original': 0, 'Deepfakes': 1, 'FaceShifter': 2, 'FaceSwap': 3, 'NeuralTextures': 4, 'Face2Face': 5, 'unknown': 6}
true_method = [method_dict.get(m, 6) for m in method_list]
pred_method = method_pred_list

print("\n================ Test Report ================")
print_confusion_matrix(true_bin, pred_bin)
print_confusion_matrix_method(true_method, pred_method)

# ✅ ROC Curve 및 t-SNE 시각화
plot_roc_curve(torch.tensor(true_bin), torch.tensor(video_bin_scores), checkpoint_path, f"{checkpoint_name}")
plot_tsne(np.array(video_feature_array), true_method, checkpoint_path, f"{checkpoint_name}", 
          method_labels=['original', 'Deepfakes', 'FaceShifter', 'FaceSwap', 'NeuralTextures', 'Face2Face', 'unknown'])
