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
import sys

# # 서버환경
# input_file_path='/root/jiyeong/Dataset/ff++/train/*'
# input_file_path2='/root/jiyeong/Dataset/DFDC/train/*'

#로컬 환경
input_file_path=f'/Users/jiyeong/Desktop/컴공 캡스톤/Dataset/ff++(원본)/NeuralTextures'
frame_file_path=f'/Users/jiyeong/Desktop/컴공 캡스톤/Dataset/ff++(원본)'
frames=150

data_list=[]

sys.stdout.reconfigure(line_buffering=True)  # 모든 print문에 flush=true 설정 반영

# 1. THis code is to check if the video is corrupted or not / 손상된 파일인지 확인 (파일 손상 시 삭제)
def validate_video(vid_path,train_transforms):
      transform = train_transforms
      count = 20
      video_path = vid_path
      frames = []
      a = int(100/count)
      first_frame = np.random.randint(0,a)
      temp_video = video_path.split('/')[-1]
      for i,frame in enumerate(frame_extract(video_path)):
        frames.append(transform(frame))
        if(len(frames) == count):
          break
      frames = torch.stack(frames)
      frames = frames[:count]
      return frames


#extract a from from video / 영상에서 프레임 추출
def frame_extract(path):
  vidObj = cv2.VideoCapture(path) 
  success = 1
  while success:
      success, image = vidObj.read()
      if success:
          yield image


im_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])



video_fil = glob.glob(f'{input_file_path}/*.mp4')  # 경로 변경
data_list.append(input_file_path)
# video_fil += glob.glob(f'{input_file_path2}/*.mp4') 
# data_list.append(input_file_path2)

print("Total no of videos :" , len(video_fil))
# print(data_list)

count = 0
for i in video_fil:
  try:
    print(i)
    count+=1
    validate_video(i,train_transforms)
  except:
    print("Number of video processed: " , count ," Remaining : " , (len(video_fil) - count))
    print("Corrupted video is : " , i)
    continue
print("no of validated_video : ",count)
print("corrupted no of video : ",(len(video_fil) - count))

frame_count = []
short_frame=[]

for video_file in reversed(video_fil): # 이거 앞에서 부터 하면 remove로 인해 frame_count랑 video_files 길이가 달라짐, 그래서 reversed 추가하여 뒤에서 부터 탐색!!
  cap = cv2.VideoCapture(video_file)
  if(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))<frames):  # frames 변수 위에서 조정
    video_fil.remove(video_file)
    short_frame.append(video_file)
    continue

  frame_count.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

# 리스트를 pandas DataFrame으로 변환
df = pd.DataFrame(video_fil, columns=['video_file'])
df2 =pd.DataFrame(short_frame, columns=['video_file'])
# 엑셀 파일로 저장
df.to_excel(f'{frame_file_path}/validate_video_files.xlsx', index=False)
df2.to_excel(f'{frame_file_path}/delete_video_files.xlsx', index=False)
print("엑셀 파일로 저장 완료!")


# print("frames are " , frame_count)
print("Total no of video: " , len(frame_count))
print('Average frame per video:',np.mean(frame_count))
print('Short_frame_count : ', len(short_frame))


