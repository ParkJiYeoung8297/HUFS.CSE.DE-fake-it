### 전체 통합 meta_data랑 global_metadata 생성

import glob
import cv2
import os
import pandas as pd


video_files =  glob.glob('/root/jiyeong/Dataset/*/*/*/*.mp4')   # 경로 지정
base_path = '/root/jiyeong/Dataset' # 기준 경로

# 저장할 엑셀 파일 경로
output_excel_path = '/root/jiyeong/Dataset/global_meta_data.xlsx'
output_csv_path = '/root/jiyeong/Dataset/Global_metadata.csv'

# 데이터 수집
file_name_list = []
folder_path_list = []
label_list = []
split_list = []
dataset_list = []
frame_count_list = []

for video_file in video_files:
    cap = cv2.VideoCapture(video_file)
    
    # 파일 이름
    file_name = os.path.basename(video_file)
    file_name_list.append(file_name)

    # 상대 경로
    relative_path = os.path.relpath(video_file, base_path).replace("\\", "/")
    folder_path_list.append(relative_path)

    # label (real/fake)
    if 'real' in relative_path.lower():
        label = 'REAL'
    elif 'fake' in relative_path.lower():
        label = 'FAKE'
    else:
        label = 'unknown'
    label_list.append(label)

    # split (train/val)
    if '/train/' in relative_path.lower():
        split = 'train'
    elif '/val/' in relative_path.lower():
        split = 'val'
    else:
        split = 'unknown'
    split_list.append(split)

    # dataset (celeb/dfdc)
    if 'fakeavceleb' in relative_path.lower():
        dataset = 'fakeavceleb'
    elif 'dfdc' in relative_path.lower():
        dataset = 'dfdc'
    elif 'celeb'in relative_path.lower():
        dataset = 'celeb'
    elif 'ff++' in relative_path.lower():
        dataset = 'ff++'
    elif 'deepspeak' in relative_path.lower():
        dataset = 'deepspeak'
    else:
        dataset = 'unknown'
    dataset_list.append(dataset)

    # frame 수
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_list.append(frame_count)

    cap.release()

# 데이터프레임 생성
df = pd.DataFrame({
    'file_name': file_name_list,
    'folder_path': folder_path_list,
    'label': label_list,
    'split': split_list,
    'dataset': dataset_list,
    'frame': frame_count_list
})

df2 = pd.DataFrame({
    'file_name': file_name_list,
    'label': label_list,
})



# 엑셀로 저장
df.to_excel(output_excel_path, index=False, engine='openpyxl')
print(f"✅ 메타데이터 생성 완료: {output_excel_path}")

# CSV로 저장
df2.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

print(f"✅ 파일 정보가 CSV로 저장되었습니다: {output_csv_path}")
