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
# frames=100
# num_epochs=int(sys.argv[5]) # 에폭 횟수

#로컬 환경
import sys
selected_model = sys.argv[1]
use_input1= int(sys.argv[2])  # 첫 번째 인자 [0,1] / 1이면 사용, 0이면 사용 X
use_input2= int(sys.argv[3])  # 두 번째 인자 [0,1]
input_file_path=f'/Users/jiyeong/Desktop/컴공 캡스톤/Dataset/ff++/train/*'
input_file_path2=f'/Users/jiyeong/Desktop/컴공 캡스톤/Dataset/DFDC/train/*'
checkpoint_name=sys.argv[4]
meta_data_path=f'/Users/jiyeong/Desktop/컴공 캡스톤/Dataset'
checkpoint_path=f'/Users/jiyeong/HUFS.CSE.DE-fake-it/model/checkpoints'
frames=100
num_epochs=int(sys.argv[5]) # 에폭 횟수
base_path='/Users/jiyeong/Desktop/컴공 캡스톤/Dataset/'

sys.stdout.reconfigure(line_buffering=True)  # 모든 print문에 flush=true 설정 반영

print("Check parameter")
print(f"model_name : {selected_model}")
print(f"ff++: {'use' if use_input1 == 1 else 'not use'}")
print(f"dfdc: {'use' if use_input2 == 1 else 'not use'}")
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
import face_recognition
import json
import copy
import random
import time

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

data_list=[]

# # 이 코드는 원격서버에서 너무 오래 걸려서 생략
# # 1. THis code is to check if the video is corrupted or not / 손상된 파일인지 확인 (파일 손상 시 삭제)
# def validate_video(vid_path,train_transforms):
#       transform = train_transforms
#       count = 20
#       video_path = vid_path
#       frames = []
#       a = int(100/count)
#       first_frame = np.random.randint(0,a)
#       temp_video = video_path.split('/')[-1]
#       for i,frame in enumerate(frame_extract(video_path)):
#         frames.append(transform(frame))
#         if(len(frames) == count):
#           break
#       frames = torch.stack(frames)
#       frames = frames[:count]
#       return frames

#extract a from from video / 영상에서 프레임 추출
# def frame_extract(path):
#   vidObj = cv2.VideoCapture(path) 
#   success = 1
#   while success:
#       success, image = vidObj.read()
#       if success:
#           yield image

# im_size = 112
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

# train_transforms = transforms.Compose([
#                                         transforms.ToPILImage(),
#                                         transforms.Resize((im_size,im_size)),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize(mean,std)])

# if use_input1==1 and use_input2==1:
#   video_fil = glob.glob(f'{input_file_path}/*.mp4')  # 경로 변경
#   data_list.append(input_file_path)
#   video_fil += glob.glob(f'{input_file_path2}/*.mp4') 
#   data_list.append(input_file_path2)
# elif use_input1==1:
#   video_fil = glob.glob(f'{input_file_path}/*.mp4')  # 경로 변경
#   data_list.append(input_file_path)
# elif use_input2==1:
#   video_fil = glob.glob(f'{input_file_path2}/*.mp4') 
#   data_list.append(input_file_path2)
# # video_fil += glob.glob('/content/drive/My Drive/DFDC_REAL_Face_only_data/*.mp4')
# print("Total no of videos :" , len(video_fil))
# # print(video_fil)
# count = 0
# for i in video_fil:
#   try:
#     count+=1
#     validate_video(i,train_transforms)
#   except:
#     print("Number of video processed: " , count ," Remaining : " , (len(video_fil) - count))
#     print("Corrupted video is : " , i)
#     continue
# print((len(video_fil) - count))


#2. to load preprocessod video to memory / 전처리된 영상 가져오기
# if use_input1==1 and use_input2==1:
#    video_files = glob.glob(f'{input_file_path}/*.mp4') 
#    video_files += glob.glob(f'{input_file_path2}/*.mp4')  
# elif use_input1==1:
#   video_files = glob.glob(f'{input_file_path}/*.mp4') 
# elif use_input2==1:
#   video_files = glob.glob(f'{input_file_path2}/*.mp4')    
# # video_files += glob.glob('/content/drive/My Drive/DFDC_FAKE_Face_only_data/*.mp4')
# random.shuffle(video_files)
# random.shuffle(video_files)

# frame_count = []
# short_frame=[]
# print(len(video_files))

# for video_file in reversed(video_files): # 이거 앞에서 부터 하면 remove로 인해 frame_count랑 video_files 길이가 달라짐, 그래서 reversed 추가하여 뒤에서 부터 탐색!!
#   cap = cv2.VideoCapture(video_file)
#   if(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))<frames):  # frames 변수 위에서 조정
#     video_files.remove(video_file)
#     short_frame.append(video_file)
#     continue

#   frame_count.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
  
# print("frames are " , frame_count)

# print("Total no of video: " , len(frame_count))
# print('Average frame per video:',np.mean(frame_count))
# print('Short_frame_count : ', len(short_frame))


# 엑셀 파일에서 데이터 읽기
df_name= pd.read_excel(f'{meta_data_path}/global_meta_data.xlsx')
if use_input1==1 and use_input2==1:
  video_files_from_excel= df_name[df_name['split'] == 'train']['folder_path'].tolist()
  video_files= [base_path+file for file in video_files_from_excel]
  data_list.append(input_file_path)
  data_list.append(input_file_path2)
elif use_input1==1:
  video_files_from_excel= df_name[(df_name['split'] == 'train') & (df_name['dataset'] == 'ff++')]['folder_path'].tolist()
  video_files= [base_path+file for file in video_files_from_excel]
  data_list.append(input_file_path)
elif use_input2==1:
  # video_files = glob.glob(f'{input_file_path2}/*.mp4') 
  video_files_from_excel= df_name[(df_name['split'] == 'train') & (df_name['dataset'] == 'dfdc')]['folder_path'].tolist()
  video_files= [base_path+file for file in video_files_from_excel]
  data_list.append(input_file_path2)

print("total no of videos : ", len(video_files))

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
        a = int(100/self.count)
        first_frame = np.random.randint(0,a)
        temp_video = video_path.split('/')[-1]
        #print(temp_video)
        label = self.labels.iloc[(labels.loc[labels["file"] == temp_video].index.values[0]),1]
        if(label == 'FAKE'):
          label = 0
        if(label == 'REAL'):
          label = 1
        for i,frame in enumerate(self.frame_extract(video_path)):
          frames.append(self.transform(frame))
          if(len(frames) == self.count):
            break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        #print("length:" , len(frames), "label",label)
        return frames,label
    
    def frame_extract(self,path):
      vidObj = cv2.VideoCapture(path) 
      success = 1
      while success:
          success, image = vidObj.read()
          if success:
              yield image

#plot the image
def im_plot(tensor):
    image = tensor.cpu().numpy().transpose(1,2,0)
    b,g,r = cv2.split(image)
    image = cv2.merge((r,g,b))
    image = image*[0.22803, 0.22145, 0.216989] +  [0.43216, 0.394666, 0.37645]
    image = image*255.0
    plt.imshow(image.astype(int))
    plt.show()


#count the number of fake and real videos
def number_of_real_and_fake_videos(data_list):
  header_list = ["file","label"]
  lab = pd.read_csv(f'{meta_data_path}/Global_metadata.csv',names=header_list)
  fake = 0
  real = 0
  for i in data_list:
    temp_video = i.split('/')[-1]
    label = lab.iloc[(labels.loc[labels["file"] == temp_video].index.values[0]),1]
    if(label == 'FAKE'):
      fake+=1
    if(label == 'REAL'):
      real+=1
  return real,fake


# load the labels and video in data loader
import random
import pandas as pd
from sklearn.model_selection import train_test_split

header_list = ["file","label"]
labels = pd.read_csv(f'{meta_data_path}/Global_metadata.csv',names=header_list)
#print(labels)

train_videos = video_files[:int(0.7*len(video_files))]  # 8:2으로 train:test
valid_videos = video_files[int(0.7*len(video_files)):]
print("train : " , len(train_videos))
print("test : " , len(valid_videos))
# train_videos,valid_videos = train_test_split(data,test_size = 0.2)
# print(train_videos)

print("TRAIN: ", "Real:",number_of_real_and_fake_videos(train_videos)[0]," Fake:",number_of_real_and_fake_videos(train_videos)[1])
print("TEST: ", "Real:",number_of_real_and_fake_videos(valid_videos)[0]," Fake:",number_of_real_and_fake_videos(valid_videos)[1])


im_size = 112
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
  train_loader = DataLoader(train_data,batch_size = 32,shuffle = True,num_workers = 2,pin_memory=True)
  valid_loader = DataLoader(val_data,batch_size = 32,shuffle = True,num_workers = 2,pin_memory=True)
else:
# cpu사용하기 때문에 병렬처리 뻄
  train_loader = DataLoader(train_data,batch_size = 32,shuffle = True,num_workers = 0)  # 여기서 batch size 조정 (한번에 몇개의 데이터를 묶어서 학습할지, batch개수=데이터 수/batch size)
  valid_loader = DataLoader(val_data,batch_size = 32,shuffle = True,num_workers = 0)
image,label = train_data[0]
# im_plot(image[0,:,:,:])

#Model with feature visualization
from torch import nn
from torchvision import models

import timm
# from efficientnet_pytorch import EfficientNet
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

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
           #  model = EfficientNet.from_pretrained('efficientnet-b0')
           #  self.model = model.extract_features
           weights = EfficientNet_B0_Weights.DEFAULT
           model = efficientnet_b0(weights=weights)
           self.model = nn.Sequential(*list(model.features))



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
    

# 모델을 device로 보내기
# model = Model(2).to(device)

model = Model(num_classes=2, model_name=selected_model).to(device)

# 입력 텐서도 device로 보내기
input_tensor = torch.from_numpy(np.empty((1, 20, 3, 112, 112))).type(torch.FloatTensor).to(device)

# 모델 실행
a, b = model(input_tensor)
    

from torch.autograd import Variable
import sys
def train_epoch(epoch, num_epochs, data_loader, model, criterion, optimizer):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    t = []

    for i, (inputs, targets) in enumerate(data_loader):
        # GPU에서 실행
        # inputs, targets device로 올리기
        inputs = inputs.to(device)
        targets = targets.to(device)

        _,outputs = model(inputs)
        # gpu에서 실행
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d / %d] [Loss: %f, Acc: %.2f%%]"
                % (
                    epoch,
                    num_epochs,
                    i,
                    len(data_loader),
                    losses.avg,
                    accuracies.avg))

    # 모델 저장
    os.makedirs(checkpoint_path, exist_ok=True)  # 폴더 없으면 생성
    # torch.save(model.state_dict(),'/content/checkpoint.pt')
    torch.save(model.state_dict(), f'{checkpoint_path}/{checkpoint_name}.pt')

    return losses.avg,accuracies.avg

def test(epoch,model, data_loader ,criterion):
    print('Testing')
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    pred = []
    true = []
    count = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            # device 선택 mps/cuda/cpu
            device = get_device()
            model = model.to(device)
            inputs = inputs.to(device)
            targets = targets.to(device)
          

            _,outputs = model(inputs)
            # GPu cuda 사용
            loss = torch.mean(criterion(outputs, targets))
            acc = calculate_accuracy(outputs, targets)
            #
            _,p = torch.max(outputs,1) 
            # true += (targets.type(torch.cuda.LongTensor)).detach().cpu().numpy().reshape(len(targets)).tolist()
            true += (targets.type(torch.LongTensor)).detach().cpu().numpy().reshape(len(targets)).tolist()
            pred += p.detach().cpu().numpy().reshape(len(p)).tolist()
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            sys.stdout.write(
                    "\r[Batch %d / %d]  [Loss: %f, Acc: %.2f%%]"
                    % (
                        i,
                        len(data_loader),
                        losses.avg,
                        accuracies.avg
                        )
                    )
        print('\nAccuracy {}'.format(accuracies.avg))
    return true,pred,losses.avg,accuracies.avg
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
def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return 100* n_correct_elems / batch_size

#Output confusion matrix / 모델 성능 평가
import seaborn as sn
from sklearn.metrics import confusion_matrix  #내가 추가함
from sklearn.metrics import classification_report, confusion_matrix
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

optimizer = torch.optim.Adam(model.parameters(), lr= lr,weight_decay = 1e-5)
criterion = nn.CrossEntropyLoss().to(device) # device에 따라 사용
#
train_loss_avg =[]
train_accuracy = []
test_loss_avg = []
test_accuracy = []

# 시간 측정 시작
start_time = time.time()
for epoch in range(1,num_epochs+1):
    epoch_start_time = time.time()
    l, acc = train_epoch(epoch,num_epochs,train_loader,model,criterion,optimizer)
    train_loss_avg.append(l)
    train_accuracy.append(acc)
    true,pred,tl,t_acc = test(epoch,model,valid_loader,criterion)
    test_loss_avg.append(tl)
    test_accuracy.append(t_acc)
    
    epoch_end_time = time.time()
    epoch_elapsed = epoch_end_time - epoch_start_time
    print(f"✅ Epoch {epoch} 소요 시간: {epoch_elapsed:.2f}초")
# 시간 측정 끝
end_time = time.time()
    
plot_loss(train_loss_avg,test_loss_avg,len(train_loss_avg))
plot_accuracy(train_accuracy,test_accuracy,len(train_accuracy))
print(confusion_matrix(true,pred))
elapsed_time = end_time - start_time
print(f"✅ 전체 학습 소요 시간: {elapsed_time:.2f}초")


print("--------------------------------------------------------Report---------------------------------------------")
print(f"✅ Using device: {device}")
print(f"✅ 전체 학습 소요 시간: {elapsed_time:.2f}초")
print(f'lr = {lr}, epoch = {num_epochs}')
print(f"사용 데이터 목록 : {data_list}" )
print_confusion_matrix(true,pred)

