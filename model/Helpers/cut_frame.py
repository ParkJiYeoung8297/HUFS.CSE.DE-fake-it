import cv2
import os
import glob

# 영상 경로와 저장할 디렉토리 설정
input_file_path = '/Users/jiyeong/Desktop/컴공 캡스톤/Dataset/ff++(원본)/train/fake/NeuralTextures'  # 영상 파일 경로
output_dir = '/Users/jiyeong/Desktop/컴공 캡스톤/Dataset/ff++/train/fake/NeuralTextures'    # 비디오 저장 디렉토리

# 저장할 디렉토리가 없다면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 영상 파일 목록 가져오기
video_files = glob.glob(f'{input_file_path}/*.mp4')  # 경로 변경
print("Total number of videos:", len(video_files))

# 각 비디오 파일에 대해 처리
for video_path in video_files:
    # 영상 파일 열기
    cap = cv2.VideoCapture(video_path)

    # 영상이 열렸는지 확인
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        continue

    # 비디오의 프레임 크기와 FPS 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # MJPG 포맷으로 저장할 VideoWriter 객체 생성
    video_name = os.path.basename(video_path).split('.')[0]
    output_video_path = os.path.join(output_dir, f"{video_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # MJPG 코덱 사용
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width,frame_height))

    # 150프레임만 읽어와서 저장
    frame_count = 0
    while frame_count < 150:

        ret, frame = cap.read()

        # 영상이 끝났으면 종료
        if not ret:
            break
        # if frame_count>=300:
        #     # 150프레임만 저장
        out.write(frame)
        frame_count += 1

    # 영상 파일 닫기
    cap.release()
    out.release()

    print(f"Saved 150 frames of {video_name} to {output_video_path}")

print("Finished saving videos.")
