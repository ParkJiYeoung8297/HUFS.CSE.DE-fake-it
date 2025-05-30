input_file_path='/Users/jiyeong/Desktop/컴공 캡스톤/FakeAVCeleb_v1.2/Dataset'
output_file_path='/Users/jiyeong/Desktop/컴공 캡스톤/FakeAVCeleb_v1.2/Dataset/fake_pr/Caucasian (American)'
# 도커 서버로 돌리기 위해 상대 주소로 변경
# input_file_path = '/input'
# output_file_path = '/output'

# 1. To get the average frame count and select the frame with 150 frames and more  - 각 영상의 프레임 수 확인 후 150 프레임 이상인 프레임만 선별
import json
import glob
import numpy as np
import cv2
import copy
import pandas as pd
def average_frame_count(i):
  input_path = f'{input_file_path}/{i}/*.mp4'  #Input file path, 입력 파일 경로 - 파일 경로 수정!!
  video_files = glob.glob(input_path)
  frame_count = []
  video_list = []
  short_frame=[]
  short_frame_count=[]
  for video_file in video_files:
    cap = cv2.VideoCapture(video_file)
    if(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))<100):
      short_frame.append(video_file)
      short_frame_count.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
      #video_files.remove(video_file) # 삭제 대신 리스트에 추가하여 목록 관리
      continue
    video_list.append(video_file) # 프레임 100 이상인 영상들
    frame_count.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()  # 자원 해제

  # print("frames" , frame_count)
  print(i)
  print("Total number of videos: " , len(frame_count))
  print('Average frame per video:',np.mean(frame_count))
  print('Short frame video:',len(short_frame))
  # print("Short frame video list:",short_frame)

  return video_list



# 2. to extract frame from video - 비디오 파일에서 프레임 추출
def frame_extract(path):
  vidObj = cv2.VideoCapture(path)
  success = 1
  while success:
      success, image = vidObj.read()
      if success:
          yield image

# 3. process the frames - 프레임에서 추출된 얼굴을 새로운 영상으로 저장
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
from tqdm.autonotebook import tqdm
# process the frames
def create_face_videos(path_list,out_dir):

  # Ensure to use MPS for MacBook -MPS GPu 사용하기
  device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
  print(f"Using device: {device}")

  # 이미 처리되어 저장된 영상 개수 확인
  already_present_count = glob.glob(out_dir + '/*.mp4')
  print("No of videos already present ", len(already_present_count))

  for path in tqdm(path_list):
    print(path)
    out_path = os.path.join(out_dir,path.split('/')[-1]) # 영상 파일 이름 추출
    print(out_path)
    file_exists = glob.glob(out_path)
    print(file_exists)
    if(len(file_exists) != 0): # 이미 존재하면 pass
      print("File Already exists: " , out_path)
      continue

    frames = []
    flag = 0
    face_all = []
    frames1 = []
    out = cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (112,112))
    for idx,frame in enumerate(frame_extract(path)):
      #if(idx % 3 == 0):
      if(idx <= 150):
        frames.append(frame)
        # 4프레임씩 묶어서 얼굴 탐지 (속도 개선 목적)
        if(len(frames) == 4):
          faces = face_recognition.batch_face_locations(frames) #얼굴 위치
          for i,face in enumerate(faces):
            if(len(face) != 0): #얼굴이 포착되면
              top,right,bottom,left = face[0] #얼굴 첫 좌표 추출
            try:
              out.write(cv2.resize(frames[i][top:bottom,left:right,:],(112,112)))
            except:
              pass
          frames = []
    try:
      del top,right,bottom,left
    except:
      pass
    out.release()


# main function (include 1,2, 3 function) - 위에 1,2,3 함수들 포함하는 메인 함수
from multiprocessing import Process  
def run_job(i, output_dir):
    video_files = average_frame_count(i)
    create_face_videos(video_files, output_dir)

# parallel processing process - 병렬 처리 프로세스
if __name__ == '__main__':
    output_path = f"{output_file_path}/men"  # Output file path - 목적지 파일 경로 , 파일 경로 수정!!
    output_path2 = f"{output_file_path}/women"

    # 각 작업을 별도 프로세스로 실행
    p1 = Process(target=run_job, args=('FakeVideo-FakeAudio/Caucasian (American)/men/*', output_path))        # Input file path, 입력 파일 경로 - 파일 경로 수정!!
    p2 = Process(target=run_job, args=('FakeVideo-FakeAudio/Caucasian (American)/women/*', output_path2))        # Input file path, 입력 파일 경로 - 파일 경로 수정!!
    # p3 = Process(target=run_job, args=('to3', output_path3))        # Input file path, 입력 파일 경로 - 파일 경로 수정!!

    # 병렬 실행 시작
    p1.start()
    p2.start()
    # p3.start()

    # 메인 프로세스는 세 개 작업 모두 완료될 때까지 대기
    p1.join()
    p2.join()
    # p3.join()
