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
import face_alignment
from tqdm import tqdm
from django.conf import settings
from pathlib import Path
import subprocess
import cv2
import json
import requests

checkpoint_path=Path(__file__).resolve().parent


# ✅ 모델 정의
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


# ✅ Grad-CAM 계산 함수
def run_grad_cam(model, input_tensor, target_class=None, device = torch.device("mps")):
    model.eval()
    fmap = None
    grad = None

    # forward hook: 마지막 layer 출력 저장
    def fw_hook(module, inp, out):
        nonlocal fmap
        fmap = out.detach()
        
    # backward hook: 마지막 layer의 gradient 저장
    def bw_hook(module, grad_in, grad_out):
        nonlocal grad
        grad = grad_out[0].detach()

    last_layer = model.model[-1]
    f = last_layer.register_forward_hook(fw_hook)
    b = last_layer.register_backward_hook(bw_hook)

    input_tensor = input_tensor.to(device).unsqueeze(0).unsqueeze(0).requires_grad_(True)
    _, output = model(input_tensor)

    pred_class = output.argmax(dim=1).item()  # 🟢 예측 클래스

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, target_class].backward()

    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam = (weights * fmap).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))

    f.remove()
    b.remove()

    return cam, pred_class


def save_Top10_gradcam_images(input_video_path, model,file_name,device):

    # transform = T.Compose([
    #     T.ToTensor(),
    #     T.Resize((224,224)),
    #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    transform = T.Compose([
        T.ToPILImage(),  # <--- 이 줄을 추가합니다!
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(f'{input_video_path}/{file_name}')
    frame_count = 0

    frame_scores = []  # 점수를 기록할 리스트
    frame_images = []  # 프레임 이미지를 저장할 리스트

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original = frame.copy()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = transform(img).to(device)

        cam , pred_class= run_grad_cam(model, input_tensor=img, device=device)  # Grad-CAM 실행
        if pred_class != 1:
            continue
        # 각 프레임의 활성도 점수 평균을 계산해서 frame_scores에 저장
        score = np.mean(cam)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        overlay = 0.4 * heatmap + 0.6 * original
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # 저장 리스트에 overlay와 original 둘 다 저장
        frame_images.append((frame_count, original,overlay))

        frame_scores.append((frame_count, score))
        # frame_images.append((frame_count, overlay))  # Grad-CAM 오버레이 결과 저장

        # # frame_scores.append((frame_count, score))
        # # frame_images.append((frame_count, original))  # 원본 프레임 이미지를 저장

        frame_count += 1

    # Grad-CAM 점수를 기준으로 상위 10개 프레임을 저장
    frame_scores.sort(key=lambda x: x[1], reverse=True)  # 점수 기준으로 정렬
    top_10_indices = [idx for idx, score in frame_scores]  # 상위 10개 인덱스 추출

    output_folder = os.path.join(input_video_path, f'gradcam_output')
    os.makedirs(output_folder, exist_ok=True)

    ori_output_folder = os.path.join(input_video_path, f'gradcam_output/original')
    os.makedirs(ori_output_folder, exist_ok=True)

    for rank, idx in enumerate(top_10_indices):
        # 순위(rank)는 상위 10개 인덱스에 대한 순서입니다.
        original_img = frame_images[idx][1]  # top_10_indices에 해당하는 원본 프레임 이미지를 가져옵니다.
        grad_img = frame_images[idx][2]  # top_10_indices에 해당하는 프레임 이미지를 가져옵니다.
        score = frame_scores[rank][1]  # 해당 프레임의 점수
        top_frame_path = os.path.join(ori_output_folder, f"frame{idx}_TOP{rank + 1}_Score{score:.4f}.jpg")
        cv2.imwrite(top_frame_path, original_img)
        top_frame_path = os.path.join(output_folder, f"frame{idx}_TOP{rank + 1}_Score{score:.4f}.jpg")
        cv2.imwrite(top_frame_path, grad_img)


    cap.release()
    return output_folder


def get_bbox(pts):
    x, y = pts[:,0], pts[:,1]
    return int(x.min()), int(y.min()), int(x.max()), int(y.max())

def roi_activation(cam, bbox, topk=0.1):
    x1, y1, x2, y2 = bbox
    patch = cam[y1:y2, x1:x2]
    mean_val = float(patch.mean())

    if np.isnan(mean_val):
        return -1
    return mean_val

    # x1, y1, x2, y2 = bbox
    # patch = cam[y1:y2, x1:x2]
    # if patch.size == 0:
    #     return -1

    # mean_val = float(patch.mean())
    # max_val = float(patch.max())
    # return 0.8 * mean_val + 0.2 * max_val 


def get_gradcam_peak(cam):
    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    return x, y

def is_point_in_bbox(x, y, bbox):
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2

def find_roi_by_distance(cam, bbox_dict):
    peak_x, peak_y = get_gradcam_peak(cam)

    distances = []
    for region, bbox in bbox_dict.items():
        cx, cy = get_center(bbox)
        dist = ((peak_x - cx) ** 2 + (peak_y - cy) ** 2) ** 0.5
        distances.append((region, dist))

    # 거리순으로 정렬
    distances.sort(key=lambda x: x[1])
    closest_roi = distances[0][0]
    second_closest_roi = distances[1][0]

    return closest_roi, second_closest_roi, (peak_x, peak_y)


def all_calculate_roi_scores(video_path,file_name,checkpoint_name='checkpoint_1'):

    # ✅ MPS 디바이스 설정
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # 모델 구조를 다시 정의
    model = Model(num_classes=2).to(device)
    model.load_state_dict(torch.load(f'{checkpoint_path}/{checkpoint_name}.pt'))
    model.eval()
    grad_cam_path=save_Top10_gradcam_images(video_path, model,file_name,device=device)
    # ------------------------------
    # ✅ 배치 처리 루프
    # ------------------------------

    fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    device=str(device)
    )


    transform = T.Compose([
        # T.ToTensor(),
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    result=[]

    grad_videos = glob.glob(f"{grad_cam_path}/*.jpg")
    output_dir_box_original = os.path.join(video_path, 'output_box_images/original')
    os.makedirs(output_dir_box_original, exist_ok=True)

    # 박스 친 결과 저장
    output_dir_box = os.path.join(video_path, 'output_box_images')
    os.makedirs(output_dir_box, exist_ok=True)

    for img_path in tqdm(grad_videos):
        original = cv2.imread(img_path)
        rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        landmarks = fa.get_landmarks(rgb)
        if not landmarks:
            # 얼굴이 감지되지 않으면 패스
            continue
        lm = landmarks[0]

        # 부위별 박스 계산 (dlib/face_alignment 68점 기준)
        jl_bbox = get_bbox(lm[0:17])
        le_bbox = get_bbox(lm[36:42])
        re_bbox = get_bbox(lm[42:48])
        le_brow_bbox = get_bbox(lm[17:22])
        re_brow_bbox = get_bbox(lm[22:27])
        nose_bbox = get_bbox(lm[27:36])
        mouth_bbox = get_bbox(lm[48:68])

        # Grad-CAM 결과 받기
        cam = cv2.imread(os.path.join(grad_cam_path, os.path.basename(img_path)),cv2.IMREAD_GRAYSCALE)  # Grad-CAM 이미지 로드
        cam = cv2.resize(cam, (original.shape[1], original.shape[0]))


        # 부위별 활성도 채점
        scores = {
            'jawline':  roi_activation(cam, jl_bbox ),
            'left_eye':  roi_activation(cam, le_bbox),
            'right_eye': roi_activation(cam, re_bbox),
            'left_eye_brow': roi_activation(cam, le_brow_bbox),
            'right_eye_brow': roi_activation(cam, re_brow_bbox),
            'nose':      roi_activation(cam, nose_bbox),
            'mouth':     roi_activation(cam, mouth_bbox)
        }
        # most_activated = max(scores, key=scores.get)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)# 점수 정렬(null은 맨 뒤로)
        first_activated = sorted_scores[0][0]   # 가장 큰 key
        second_activated = sorted_scores[1][0]
        # 최고 활성 부위 박스 가져오기
        bbox_map = {
            'jawline': jl_bbox,
            'left_eye':  le_bbox,
            'right_eye': re_bbox,
            'left_eye_brow':le_brow_bbox,
            'right_eye_brow':re_brow_bbox,
            'nose':      nose_bbox,
            'mouth':     mouth_bbox
        }
        f_x1, f_y1, f_x2, f_y2 = bbox_map[first_activated]
        s_x1, s_y1, s_x2, s_y2 = bbox_map[second_activated]

        # Grad-CAM peak 좌표 및 가장 가까운 ROI 두 개
        peak_roi, second_peak_roi, (px, py) = find_roi_by_distance(cam, bbox_map)

        # 결과 오버레이
        cv2.rectangle(original, (f_x1, f_y1), (f_x2, f_y2), (0, 255, 0), 2)  # Green
        cv2.rectangle(original, (s_x1, s_y1), (s_x2, s_y2), (255,0, 0), 2)  # Blue

        # if any(f"_TOP{i}_" in file_name for i in range(1, 11)):
        if any(f"_TOP{i}_" in os.path.basename(img_path) for i in range(1,2)):
            result.append({
                'file_name': os.path.basename(img_path),
                'first_activate': first_activated,
                'second_activate': second_activated,
                'f_x1': f_x1, 'f_y1': f_y1, 'f_x2': f_x2, 'f_y2': f_y2,
                's_x1': s_x1, 's_y1': s_y1, 's_x2': s_x2, 's_y2': s_y2,
                'jawline': scores['jawline'],
                'left_eye': scores['left_eye'],
                'right_eye': scores['right_eye'],
                'left_eye_brow': scores['left_eye_brow'],
                'right_eye_brow': scores['right_eye_brow'],
                'nose': scores['nose'],
                'mouth': scores['mouth'],
                'gradcam_peak_x': px,
                'gradcam_peak_y': py,
                'peak_roi': peak_roi,
                'peak_roi_2nd': second_peak_roi,
            })

        out_path_box = os.path.join(output_dir_box, os.path.basename(img_path))
        cv2.imwrite(out_path_box, original)
        
        # 원본 이미지에 박스
        orig_path = os.path.join(grad_cam_path, 'original', os.path.basename(img_path))
        if os.path.exists(orig_path):
            orig_img = cv2.imread(orig_path)
            cv2.rectangle(orig_img, (f_x1, f_y1), (f_x2, f_y2), (0, 255, 0), 2)
            cv2.rectangle(orig_img, (s_x1, s_y1), (s_x2, s_y2), (255, 0, 0), 2)
            out_orig_path = os.path.join(output_dir_box_original, os.path.basename(orig_path))
            cv2.imwrite(out_orig_path, orig_img)

    # ✅ 원본 위에 박스만 그린 영상 저장
    cap = cv2.VideoCapture(f'{video_path}/{file_name}')
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_video_path_original = os.path.join(video_path, f'output_box_on_original.mp4')

    orig_frames = sorted(glob.glob(f"{output_dir_box_original}/*.jpg"))

    if orig_frames:
        sample_frame = cv2.imread(orig_frames[0])
        height, width, _ = sample_frame.shape
        cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer_orig = cv2.VideoWriter(output_video_path_original, fourcc, fps, (width, height))

        for frame_path in orig_frames:
            frame = cv2.imread(frame_path)
            video_writer_orig.write(frame)

        video_writer_orig.release()


    # ✅ grad-cam 영상
    grad_video_path_original = os.path.join(video_path, f'grad_cam_on_original.mp4')
    orig_frames = sorted(glob.glob(f"{grad_cam_path}/*.jpg"))

    if orig_frames:
        sample_frame = cv2.imread(orig_frames[0])
        height, width, _ = sample_frame.shape
        cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer_orig = cv2.VideoWriter(grad_video_path_original, fourcc, fps, (width, height))

        for frame_path in orig_frames:
            frame = cv2.imread(frame_path)
            video_writer_orig.write(frame)

        video_writer_orig.release()




    # 이미 저장된 원본 영상 경로
    grad_video_path_original = os.path.join(video_path, 'grad_cam_on_original.mp4')

    # 변환 후 저장할 경로 설정
    output_filename = "converted_grad_cam_on_original.mp4"
    output_path = os.path.join(video_path, output_filename)

    # ffmpeg를 사용하여 변환 수행
    try:
        subprocess.run([
            'ffmpeg', '-i', grad_video_path_original,
            '-vcodec', 'libx264',
            '-acodec', 'aac',
            '-strict', 'experimental',
            '-y',  # 덮어쓰기
            output_path
        ], check=True)

        print(f"✅ Video converted: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during conversion: {e}")

    # 이미 저장된 원본 영상 경로
    grad_video_path_original = os.path.join(video_path, 'output_box_on_original.mp4')

    # 변환 후 저장할 경로 설정
    output_filename = "converted_output_box_on_original.mp4"
    output_path = os.path.join(video_path, output_filename)

    # ffmpeg를 사용하여 변환 수행
    try:
        subprocess.run([
            'ffmpeg', '-i', grad_video_path_original,
            '-vcodec', 'libx264',
            '-acodec', 'aac',
            '-strict', 'experimental',
            '-y',  # 덮어쓰기
            output_path
        ], check=True)

        print(f"✅ Video converted: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during conversion: {e}")



    def format_prompt(entry):
        return (
            f"해당 영상 ({entry['file_name']})은 Grad-CAM 시각화 결과 "
            f"{entry['peak_roi']} 부위에서 가장 높은 활성화가 관측되었고, "
            f"{entry['first_activate']}와 {entry['second_activate']} 부위가 순차적으로 높은 활성도를 보였습니다. "
            f"활성도 점수는 "
            f"jawline: {entry['jawline']:.4f}, left_eye: {entry['left_eye']:.4f}, right_eye: {entry['right_eye']:.4f}, "
            f"left_eye_brow: {entry['left_eye_brow']:.4f}, right_eye_brow: {entry['right_eye_brow']:.4f}, "
            f"nose: {entry['nose']:.4f}, mouth: {entry['mouth']:.4f}이며, "
            f"gradcam의 peak 위치는 ({entry['gradcam_peak_x']}, {entry['gradcam_peak_y']})입니다. "
            f"이 정보를 바탕으로 이 영상이 딥페이크로 판단된 이유를 설명해주세요. 판단 이유만 설명하면 되고, 번호 매겨서 가독성있게 문장을 줄 바꿔쓰기 기호 포함해서 주세요"
        )

    # def format_response(entry):
    #     return (
    #         f"{entry['peak_roi']}와 {entry['first_activate']}, {entry['second_activate']} 부위에서 높은 활성도가 관측되었습니다. "
    #         f"이 부위는 실제 인물의 움직임과 다른 왜곡된 패턴이 보이므로 딥페이크로 판단됩니다."
    #     )

    # with open("data.jsonl", "w", encoding="utf-8") as f:
    #     for entry in result:
    #         prompt = format_prompt(entry)
    #         response = format_response(entry)

    #         json_line = {"prompt": prompt, "response": response}
    #         f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
    def query_model(prompt):
        url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "llama3",
            "prompt": prompt
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            print("응답 내용:")
            # 응답 내용이 여러 개로 쪼개져서 오는 경우 합치기
            combined_response = ''
            
            # 서버 응답이 여러 줄로 온다면, 각 줄을 처리해서 하나의 문자열로 합침
            for line in response.text.splitlines():
                try:
                    # 각 줄을 JSON으로 파싱
                    json_line = json.loads(line)
                    # 'response' 키의 값만 합침
                    combined_response += json_line['response']
                except json.JSONDecodeError:
                    continue  # 잘못된 줄은 무시

            # 합친 응답 출력
            print(combined_response)
            return combined_response
        else:
            print("요청 실패:", response.status_code)

    response_txt=query_model(format_prompt(result[0]))
    return response_txt,grad_cam_path,output_dir_box
