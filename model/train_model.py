# # # 원격 서버 환경
# import sys
# selected_model = sys.argv[1]
# use_input1= int(sys.argv[2])  # 첫 번째 인자 [0,1] / 1이면 사용, 0이면 사용 X
# use_input2= int(sys.argv[3])  # 두 번째 인자 [0,1]
# input_file_path='/root/jiyeong/Dataset/ff++/train/*'
# input_file_path2='/root/jiyeong/Dataset/DFDC/train/*'
# base_path='/root/jiyeong/Dataset/'


# meta_data_path='/root/jiyeong/Dataset'
# checkpoint_path='/root/jiyeong/model/checkpoints'
# checkpoint_name=sys.argv[4] # 체크포인트 이름
# frames=150
# num_epochs=int(sys.argv[5]) # 에폭 횟수

# # 코랩 서버 환경
# import sys
# selected_model = sys.argv[1]
# input_file_path='/content/drive/MyDrive/Capstone/Dataset/ff++/train/*/*'
# base_path='/content/drive/MyDrive/Capstone/Dataset/'


# meta_data_path='/content/drive/MyDrive/Capstone/Dataset/ff++'
# checkpoint_path='/content/drive/MyDrive/Capstone/checkpoints'
# checkpoint_name=sys.argv[2] # 체크포인트 이름
# frames=150
# num_epochs=int(sys.argv[3]) # 에폭 횟수

# #로컬 환경
import sys
selected_model = sys.argv[1]
input_file_path=f'/Users/jiyeong/Desktop/컴공 캡스톤/Dataset/ff++/train/*/*'
checkpoint_name=sys.argv[2]
meta_data_path=f'/Users/jiyeong/Desktop/컴공 캡스톤/Dataset/ff++'
checkpoint_path=f'/Users/jiyeong/HUFS.CSE.DE-fake-it/model/checkpoints'
frames=150
num_epochs=int(sys.argv[3]) # 에폭 횟수
base_path='/Users/jiyeong/Desktop/컴공 캡스톤/Dataset/'

sys.stdout.reconfigure(line_buffering=True)  # 모든 print문에 flush=true 설정 반영

print("Check parameter")
print(f"model_name : {selected_model}")
print(f"Dataset : FaceForencis++")
print(f"Checkpoint name: {checkpoint_name}")
print(f"Training for {num_epochs} epochs")
print()

import cv2
import torch
import torchvision.transforms as T
import numpy as np
import torch.nn.functional as F
from torchvision import models
from torch import nn
import os
import glob
import pandas as pd
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
# import face_recognition
import json
import copy
import random
import time
import seaborn as sn
from torch import nn
from torchvision import models
import timm
# from efficientnet_pytorch import EfficientNet
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import confusion_matrix  #내가 추가함
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from torch.autograd import Variable
import sys

# device / 디바이스 설정
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

device = get_device()
print(f"✅ Using device: {device}")


#2. to load preprocessod video to memory / 전처리된 영상 가져오기
video_files = glob.glob(f'{input_file_path}/*.mp4') 
random.shuffle(video_files)
random.shuffle(video_files)

# 3. load the video name and labels from csv / metadata에서 real/fake 여부 가져오기
class video_dataset(Dataset):
    def __init__(self,video_names,labels,sequence_length = 60,transform = None):
        self.video_names = video_names
        self.labels = labels
        self.transform = transform
        self.count = sequence_length
    def __len__(self):
        return len(self.video_names)
    def __getitem__(self,idx):
        video_path = self.video_names[idx]
        frames = []
        # a = int(100/self.count)
        # first_frame = np.random.randint(0,a)
        temp_video = video_path.split('/')[-1]
        label = self.labels.iloc[(labels.loc[labels["file"] == temp_video].index.values[0]),1]
        if(label == 'FAKE'):
          label = 0
        if(label == 'REAL'):
          label = 1

        method_str= self.labels.iloc[(labels.loc[labels["file"] == temp_video].index.values[0]),2]

        # method를 숫자 라벨로 매핑
        method_dict = {'original': 0, 'Deepfakes': 1, 'FaceShifter': 2, 'FaceSwap': 3, 'NeuralTextures': 4, 'Face2Face':5, 'unknown': 6 }
        method = method_dict[method_str]

        for i,frame in enumerate(self.frame_extract(video_path)):
          frames.append(self.transform(frame))
          if(len(frames) == self.count):
            break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames,label,method
    
    def frame_extract(self,path):
      vidObj = cv2.VideoCapture(path) 
      success = 1
      while success:
          success, image = vidObj.read()
          if success:
              yield image

# count the number of fake and real videos / real fake 영상 개수 세기
def number_of_real_and_fake_videos(data_list):
  header_list = ["file","label","method"]
  lab = pd.read_csv(f'{meta_data_path}/Global_metadata.csv',names=header_list)
  fake = 0
  real = 0
  original=0
  deepfakes=0
  faceshifter=0
  faceswap=0
  neuraltextures=0
  face2face=0
  unknown=0
  for i in data_list:
    temp_video = i.split('/')[-1]
    label = lab.iloc[(lab.loc[labels["file"] == temp_video].index.values[0]),1]
    if(label == 'FAKE'):
      fake+=1
    if(label == 'REAL'):
      real+=1
    method = lab.iloc[(lab.loc[labels["file"] == temp_video].index.values[0]),2]
    method = method.lower()
    if(method == 'original'):
      original+=1
    elif(method == 'deepfakes'):
      deepfakes+=1
    elif(method == 'faceshifter'):
      faceshifter+=1
    elif(method == 'faceswap'):
      faceswap+=1
    elif(method == 'neuraltextures'):
      neuraltextures+=1
    elif(method == 'face2face'):
      face2face+=1
    else:
       unknown+=1
    
    
  return real,fake,original, deepfakes,faceshifter, faceswap, neuraltextures,face2face, unknown


# load the labels and video in data loader
import random
import pandas as pd
from sklearn.model_selection import train_test_split

header_list = ["file","label","method"]
labels = pd.read_csv(f'{meta_data_path}/Global_metadata.csv',names=header_list)
#print(labels)

train_videos = video_files[:int(0.9*len(video_files))]  # 8:2으로 train:test
valid_videos = video_files[int(0.9*len(video_files)):]
print("train : " , len(train_videos))
print("test : " , len(valid_videos))
# train_videos,valid_videos = train_test_split(data,test_size = 0.2)
# print(train_videos)

train_count=number_of_real_and_fake_videos(train_videos)
test_count=number_of_real_and_fake_videos(valid_videos)
print(f"TRAIN:  Real: {train_count[0]} Fake: {train_count[1]} original : {train_count[2]} Deepfakes : {train_count[3]}",
      f"FaceShifter : {train_count[4]}  FaceSwap : {train_count[5]} NeuralTextures : {train_count[6]} Face2Face {train_count[7]} unknown : {train_count[8]}")
print(f"TEST:  Real: {test_count[0]} Fake: {test_count[1]} original : {test_count[2]} Deepfakes : {test_count[3]}",
      f"FaceShifter : {test_count[4]}  FaceSwap : {test_count[5]} NeuralTextures : {test_count[6]} Face2Face {test_count[7]} unknown : {test_count[8]}")
im_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

test_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

train_data = video_dataset(train_videos,labels,sequence_length = 10,transform = train_transforms)
#print(train_data)
val_data = video_dataset(valid_videos,labels,sequence_length = 10,transform = train_transforms)


if device.type == "cuda":
  train_loader = DataLoader(train_data,batch_size = 8,shuffle = True,num_workers = 2,pin_memory=True)
  valid_loader = DataLoader(val_data,batch_size = 8,shuffle = True,num_workers = 2,pin_memory=True)
else:
# cpu사용하기 때문에 병렬처리 뻄
  train_loader = DataLoader(train_data,batch_size = 16,shuffle = True,num_workers = 0)  # 여기서 batch size 조정 (한번에 몇개의 데이터를 묶어서 학습할지, batch개수=데이터 수/batch size)
  valid_loader = DataLoader(val_data,batch_size = 16,shuffle = True,num_workers = 0)

#Model with feature visualization

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

    

# 모델을 device로 보내기
# model = Model(2).to(device)
model = Model(num_binary_classes=2, num_method_classes=7, model_name=selected_model).to(device)

# 입력 텐서도 device로 보내기
input_tensor = torch.from_numpy(np.empty((1, 20, 3, 224,224))).type(torch.FloatTensor).to(device)

# 모델 실행
fmap, output_bin, output_method = model(input_tensor)
    

def train_epoch(epoch, num_epochs, data_loader, model, criterion_bin, criterion_method, optimizer):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    t = []

    for i, (inputs,targets_bin, targets_method) in enumerate(data_loader):
        # GPU에서 실행
        # inputs, targets device로 올리기
        inputs = inputs.to(device)
        targets_bin = targets_bin.to(device)
        targets_method = targets_method.to(device)

        _, output_bin, output_method = model(inputs)
        # gpu에서 실행
        loss_bin = criterion_bin(output_bin, targets_bin)
        loss_method = criterion_method(output_method, targets_method)
        loss = loss_bin + loss_method

        acc = calculate_accuracy(output_bin, targets_bin)
        acc_method = calculate_accuracy(output_method, targets_method)
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d / %d] [Loss: %f, Acc: %.2f%%, Acc(method): %.2f%%]"
                % (
                    epoch,
                    num_epochs,
                    i,
                    len(data_loader),
                    losses.avg,
                    accuracies.avg,
                    acc_method,
                    ))

    # save the model / 모델 저장
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save(model.state_dict(), f'{checkpoint_path}/{checkpoint_name}.pt')

    return losses.avg,accuracies.avg,acc_method

def test(epoch,model, data_loader ,criterion_bin, criterion_method):
    print('Testing')
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    pred_bin= []
    true_bin= []
    pred_method = []
    true_method = []
    output_bin_all = [] # for ROC
    feature_list = [] # for t-SNE

    count = 0
    with torch.no_grad():
        for i, (inputs, targets_bin, targets_method) in enumerate(data_loader):
            # device 선택 mps/cuda/cpu
            device = get_device()
            model = model.to(device)
            inputs = inputs.to(device)
            targets_bin = targets_bin.to(device)
            targets_method = targets_method.to(device)
          

            fmap, output_bin, output_method = model(inputs)
            batch_size, seq_length = inputs.shape[0], inputs.shape[1]
            features = fmap.view(batch_size, seq_length, -1).mean(dim=1)# 평균 pooling
            feature_list.append(features.cpu().numpy())
            output_bin_all.append(output_bin.detach().cpu())
            # GPu cuda 사용
            loss_bin = criterion_bin(output_bin, targets_bin)
            loss_method = criterion_method(output_method, targets_method)
            loss = loss_bin + loss_method

            acc = calculate_accuracy(output_bin, targets_bin)
            acc_method = calculate_accuracy(output_method, targets_method)
            #
            _, p_bin = torch.max(output_bin, 1)
            _, p_method = torch.max(output_method, 1)
            # # true += (targets.type(torch.cuda.LongTensor)).detach().cpu().numpy().reshape(len(targets)).tolist()
            # true += (targets.type(torch.LongTensor)).detach().cpu().numpy().reshape(len(targets)).tolist()
            # pred += p.detach().cpu().numpy().reshape(len(p)).tolist()
            true_bin += targets_bin.cpu().numpy().tolist()
            pred_bin += p_bin.cpu().numpy().tolist()
            true_method += targets_method.cpu().numpy().tolist()
            pred_method += p_method.cpu().numpy().tolist()


            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            sys.stdout.write(
                    "\r[Batch %d / %d]  [Loss: %f, Acc: %.2f%%, Acc(method): %.2f%%]"
                    % (
                        i,
                        len(data_loader),
                        losses.avg,
                        accuracies.avg,
                        acc_method,
                        )
                    )
        print('\nAccuracy {}'.format(accuracies.avg))

    output_bin_all = torch.cat(output_bin_all, dim=0)  # [N, 2] 형태로 만듦
    feature_array = np.concatenate(feature_list, axis=0)  # [N, D] 형태
    return true_bin, pred_bin, true_method, pred_method,losses.avg,accuracies.avg,acc_method,output_bin_all,feature_array 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(outputs, targets): # top-1 accuracy
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return 100* n_correct_elems / batch_size

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
    plt.savefig(f'{checkpoint_path}/{checkpoint_name}_plot.png')
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
    plt.savefig(f'{checkpoint_path}/{checkpoint_name}_plot(method).png')

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
    pred_score = torch.softmax(output_bin, dim=1)[:, 1].cpu().numpy()  # Real 확률
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
    plt.savefig(f"{checkpoint_path}/{checkpoint_name}_roc_curve.png")
    print(f"✅ ROC Curve saved to {checkpoint_path}/{checkpoint_name}_roc_curve.png")

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
    plt.savefig(f"{checkpoint_path}/{checkpoint_name}_tsne.png")
    print(f"✅ t-SNE plot saved to {checkpoint_path}/{checkpoint_name}_tsne.png")



# loss 그래프
def plot_loss(train_loss_avg,test_loss_avg,num_epochs):
  loss_train = train_loss_avg
  loss_val = test_loss_avg
  print(num_epochs)
  epochs = range(1,num_epochs+1)
  plt.clf()  # Clear the previous figure
  plt.plot(epochs, loss_train, 'g', label='Training loss')
  plt.plot(epochs, loss_val, 'b', label='validation loss')
  plt.title('Training and Validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  # plt.show()
  plt.savefig(f'{checkpoint_path}/{checkpoint_name}_loss_plot.png')

def plot_accuracy(train_accuracy,test_accuracy,num_epochs):
  loss_train = train_accuracy
  loss_val = test_accuracy
  epochs = range(1,num_epochs+1)
  plt.clf()  # Clear the previous figure
  plt.plot(epochs, loss_train, 'g', label='Training accuracy')
  plt.plot(epochs, loss_val, 'b', label='validation accuracy')
  plt.title('Training and Validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  # plt.show()
  plt.savefig(f'{checkpoint_path}/{checkpoint_name}_accuracy_plot.png')


import time
#learning rate
lr = 1e-4              #시작 1e-5#0.001
#number of epochs (맨 위에서 설정)
#num_epochs = 2

# criterion_bin = nn.CrossEntropyLoss().to(device)
weights = torch.tensor([5.0, 1.0]).to(device)  # [fake, real]의 순서라고 가정
criterion_bin = nn.CrossEntropyLoss(weight=weights).to(device)
criterion_method = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr= lr,weight_decay = 1e-5)


# 체크포인트가 존재하면 불러오기
checkpoint_file = f'{checkpoint_path}/{checkpoint_name}.pt'
if os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"📦 체크포인트에서 학습 재개: Epoch {start_epoch}부터")
    train_loss_avg = checkpoint['train_loss']
    train_accuracy = checkpoint['train_acc']
    test_loss_avg = checkpoint['val_loss']
    test_accuracy = checkpoint['val_acc']
else:
    print("🚨 체크포인트가 없어 처음부터 시작합니다.")
    start_epoch = 1
    train_loss_avg, train_accuracy = [], []
    test_loss_avg, test_accuracy = [], []

def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"💾 체크포인트 저장 완료: {filename}")

patience = 5
best_val_loss = float('inf')
patience_counter = 0

# 시간 측정 시작
start_time = time.time()
for epoch in range(start_epoch, num_epochs + 1):
    epoch_start_time = time.time()
    l, acc ,m_acc= train_epoch(epoch,num_epochs,train_loader,model,criterion_bin, criterion_method,optimizer)
    train_loss_avg.append(l)
    train_accuracy.append(acc)
    true_bin, pred_bin, true_method, pred_method, tl, t_acc, m_acc,output_bin_all,feature_array= test(epoch,model,valid_loader,criterion_bin, criterion_method)
    test_loss_avg.append(tl)
    test_accuracy.append(t_acc)

    epoch_end_time = time.time()
    epoch_elapsed = epoch_end_time - epoch_start_time
    print(f"✅ Epoch {epoch} 소요 시간: {epoch_elapsed:.2f}초")
    save_checkpoint({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss_avg,
        'train_acc': train_accuracy,
        'val_loss': test_loss_avg,
        'val_acc': test_accuracy
    }, checkpoint_file)

    if tl < best_val_loss:
      best_val_loss = tl
      patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"🛑 조기 종료: {epoch} epoch에서 val_loss 개선 없음")
            break
# 시간 측정 끝
end_time = time.time()
    
plot_loss(train_loss_avg,test_loss_avg,len(train_loss_avg))
plot_accuracy(train_accuracy,test_accuracy,len(train_accuracy))
print(confusion_matrix(true_bin,pred_bin))
elapsed_time = end_time - start_time
print(f"✅ 전체 학습 소요 시간: {elapsed_time:.2f}초")



print("--------------------------------------------------------Report---------------------------------------------")
print(f"✅ Using device: {device}")
print(f"✅ 전체 학습 소요 시간: {elapsed_time:.2f}초")
print(f'lr = {lr}, epoch = {num_epochs}')
print_confusion_matrix(true_bin,pred_bin)     #confusion_matrix(이진 분류)
print_confusion_matrix_method(true_method, pred_method)  #confusion_matrix(다중 분류)
plot_roc_curve(true_bin, output_bin_all, checkpoint_path, checkpoint_name) # ROC Curve (이진 분류)
plot_tsne(feature_array, true_method, checkpoint_path, checkpoint_name,    # t-SNE (다중 분류)
          method_labels=['original', 'Deepfakes', 'FaceShifter', 'FaceSwap', 'NeuralTextures', 'Face2Face', 'unknown'])
