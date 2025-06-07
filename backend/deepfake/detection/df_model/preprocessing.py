import os
import cv2
import torch
import face_recognition

def process_single_video(video_path, output_path,filename ):
    # 출력 경로 생성
    os.makedirs(output_path, exist_ok=True)
    output_video_path = os.path.join(output_path,filename)

    # 이미 처리된 경우 생략
    # if os.path.exists(output_video_path):
    #     return output_video_path

    def frame_extract(path):
        vidObj = cv2.VideoCapture(path)
        success = 1

        while success:
            success, image = vidObj.read()
            if success:
                yield image

    # VideoWriter 설정
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (112,112))
    frames = []

    for idx, frame in enumerate(frame_extract(video_path)):
        # if idx > 150:
        #     break
        frames.append(frame)
        if len(frames) == 4:
            faces = face_recognition.batch_face_locations(frames)
            for i, face in enumerate(faces):
                if len(face) != 0:
                    top, right, bottom, left = face[0]
                    try:
                        cropped = cv2.resize(frames[i][top:bottom, left:right], (112,112))
                        out.write(cropped)
                    except:
                        pass
            frames = []

    out.release()
    return output_video_path
