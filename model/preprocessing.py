from facenet_pytorch import MTCNN
import cv2
import os
import glob
import tqdm
import torch
import json
import glob
import numpy as np
import copy
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
# import face_recognition
# from tqdm.autonotebook import tqdm
import tqdm

# 장치 설정
# device = torch.device("mps") if torch.backends.mps.is_available() else (
# torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# )
device=torch.device("cpu")
print(f"Using device: {device}")


def average_frame_count(input_file_path):
  input_path = f'{input_file_path}/*.mp4'  #Input file path, 입력 파일 경로 - 파일 경로 수정!!
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


# MTCNN 초기화
mtcnn = MTCNN(keep_all=False, device=device)  # 가장 잘 잡힌 얼굴 하나만 사용

def create_face_videos(path_list, out_dir):

    # 이미 처리된 영상 개수 출력
    already_present_count = glob.glob(out_dir + '/*.mp4')
    print("No of videos already present:", len(already_present_count))

    for path in tqdm.tqdm(path_list):
        out_path = os.path.join(out_dir, os.path.basename(path))

        # 이미 존재하면 건너뜀
        if os.path.exists(out_path):
            print("File already exists:", out_path)
            continue

        frames = []
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (224,224))

        for idx, frame in enumerate(frame_extract(path)):
            # if idx > 150:
            #     break

            # BGR to RGB 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 얼굴 감지
            box, _ = mtcnn.detect(rgb_frame)

            if box is not None:
                x1, y1, x2, y2 = map(int, box[0])  # 첫 번째 얼굴만 사용
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                face = frame[y1:y2, x1:x2]
                if face.size != 0:
                    resized_face = cv2.resize(face, (224,224))
                    out.write(resized_face)

        out.release()



input_file_path='/Users/jiyeong/Desktop/컴공 캡스톤/Dataset/FaceForensics++_C23'
output_file_path='/Users/jiyeong/Desktop/컴공 캡스톤/Dataset/ff++(원본)'
data_details="NeuralTextures"

# main function (include 1,2, 3 function) - 위에 1,2,3 함수들 포함하는 메인 함수
from multiprocessing import Process  
def run_job(input_dir, output_dir):
  video_files = average_frame_count(input_dir)
  create_face_videos(video_files, output_dir) 

# parallel processing process - 병렬 처리 프로세스
if __name__ == '__main__':

  input_path = f"{input_file_path}/{data_details}"
  input_path2 = f"{input_file_path}/{data_details}2"
  # input_path3 = f"{input_file_path}/{data_details}3"
  output_path = f"{output_file_path}/{data_details}"

  # 각 작업을 별도 프로세스로 실행
  p1 = Process(target=run_job, args=(input_path, output_path))        # Input file path, 입력 파일 경로
  # p2 = Process(target=run_job, args=(input_path2, output_path))        # Input file path, 입력 파일 경로 
  # p3 = Process(target=run_job, args=(input_path2, output_path))        # Input file path, 입력 파일 경로 

  # 병렬 실행 시작
  p1.start()
  # p2.start()
  # p3.start()

  # 메인 프로세스는 세 개 작업 모두 완료될 때까지 대기
  p1.join()
  # p2.join()
  # p3.join()
