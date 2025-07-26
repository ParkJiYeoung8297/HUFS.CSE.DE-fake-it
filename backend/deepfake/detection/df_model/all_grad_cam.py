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
from .model import Model

checkpoint_path=Path(__file__).resolve().parent

# ✅ Set the device to MPS(for Mac) if available, otherwise fallback to CUDA or CPU
device = torch.device("mps") if torch.backends.mps.is_available() else (
torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")




# ✅ Grad-CAM computation for binary classification
def compute_gradcam_binary(model, input_tensor, target_class=0,device = torch.device("mps")):
    fmap = None
    grad = None

    def fw_hook(module, inp, out):
        nonlocal fmap
        fmap = out.detach()

    def bw_hook(module, grad_in, grad_out):
        nonlocal grad
        grad = grad_out[0].detach()

    last_layer = model.model[-1]
    f = last_layer.register_forward_hook(fw_hook)
    b = last_layer.register_backward_hook(bw_hook)

    input_tensor = input_tensor.to(device).unsqueeze(0).unsqueeze(0).requires_grad_(True)
    _, binary_output, method_output = model(input_tensor)

    # Get the probability of the target class
    prob = F.softmax(binary_output, dim=1)[0, target_class].item()

    # Predict binary and method classes
    binary_pred = torch.argmax(binary_output, dim=1).item()   # 0: fake, 1: real
    method_pred = torch.argmax(method_output, dim=1).item()   # 0: original, 1~6: fake methods, 7: others

    # 🔴 Condition 1: If the prediction is real and the method is original, skip CAM computation / 조건 1: real(1) + original(0) → CAM X
    if binary_pred == 1 and method_pred==0:
        cam = np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]))
        f.remove()
        b.remove()
        return cam,prob,binary_pred, method_pred

    # Grad-CAM for fake class (target_class = 0)
    target_class = 0
    model.zero_grad()
    binary_output[0, target_class].backward()

    # Compute Grad-CAM
    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam = (weights * fmap).sum(dim=1, keepdim=True)
    cam = F.relu(cam).squeeze().cpu().numpy()

    # 🔵 Condition 2: If the prediction is fake and the method is not original, enhance CAM / 조건 2: fake (0) + method (1~7) (≠ 0) → CAM ↑
    if binary_pred == 0 and method_pred != 0:
        cam *=1.5

    # Normalize and resize the CAM
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))



    f.remove()
    b.remove()
    return cam, prob, binary_pred, method_pred

def get_bbox(pts):
    x, y = pts[:,0], pts[:,1]
    return int(x.min()), int(y.min()), int(x.max()), int(y.max())

def roi_activation(cam, bbox):
    x1, y1, x2, y2 = bbox
    patch = cam[y1:y2, x1:x2]
    mean_val = float(patch.mean())

    if np.isnan(mean_val):
        return -1
    return mean_val

def analyze_roi_activation(video_dir,file_name, result,model):

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device=str(device)
    )

    os.makedirs(video_dir, exist_ok=True)

    transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    roi_result = []
    roi_analyze_result={}

    # video_paths = glob.glob(os.path.join(video_path, '*.mp4'))

#   for video_path in tqdm(video_paths):

    facial_region=['jawline', 'left_eye', 'right_eye', 'left_eye_brow', 'right_eye_brow', 'nose', 'mouth','None']
    first_detection_count = {key: 0 for key in facial_region}
    second_detection_count = {key: 0 for key in facial_region}
    detection_probabillity={key: 0.0 for key in facial_region}


    video_path_file_name=os.path.join(video_dir,file_name)
    cap = cv2.VideoCapture(video_path_file_name)
    frame_idx = 0
    # video_name = os.path.splitext(os.path.basename(video_path))[0]


    # ▶️ 비디오 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 박스만 그린 영상
    video_writer_box = cv2.VideoWriter(
        os.path.join(video_dir, "output_box_on_original.mp4"),
        fourcc, fps, (width, height)
    )

    # Grad-CAM + 박스 영상
    video_writer_cam = cv2.VideoWriter(
        os.path.join(video_dir, f"grad_cam_on_original.mp4"),
        fourcc, fps, (width, height)
    )

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = fa.get_landmarks(rgb)
        if not landmarks:
            frame_idx += 1
            continue
        lm = landmarks[0]

        # ROI BBoxes
        bbox_map = {
            'jawline': get_bbox(lm[0:17]),
            'left_eye': get_bbox(lm[36:42]),
            'right_eye': get_bbox(lm[42:48]),
            'left_eye_brow': get_bbox(lm[17:22]),
            'right_eye_brow': get_bbox(lm[22:27]),
            'nose': get_bbox(lm[27:36]),
            'mouth': get_bbox(lm[48:68]),
        }

        # Grad-CAM
        img = transform(rgb).to(device)
        cam, cam_score, binary_pred, method_pred = compute_gradcam_binary(model, img)
        cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]))

        scores = {region: roi_activation(cam, box) for region, box in bbox_map.items()}
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        first_activated_region = sorted_scores[0][0]
        second_activated_region = sorted_scores[1][0]

        f_x1, f_y1, f_x2, f_y2 = bbox_map[first_activated_region]
        s_x1, s_y1, s_x2, s_y2 = bbox_map[second_activated_region]

        # 1️⃣ Grad-CAM 히트맵 오버레이
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        # 2️⃣ 원본 프레임 복사해서 박스용 준비
        frame_with_box = frame.copy()
        overlay_with_box = overlay.copy()

        # 3️⃣ 박스 그리기
        if scores[first_activated_region] > 0:
            cv2.rectangle(frame_with_box, (f_x1, f_y1), (f_x2, f_y2), (0, 255, 0), 2)
            cv2.rectangle(overlay_with_box, (f_x1, f_y1), (f_x2, f_y2), (0, 255, 0), 2)
        else:
            first_activated_region="None"

        if scores[second_activated_region] > 0:
            cv2.rectangle(frame_with_box, (s_x1, s_y1), (s_x2, s_y2), (255, 0, 0), 2)
            cv2.rectangle(overlay_with_box, (s_x1, s_y1), (s_x2, s_y2), (255, 0, 0), 2)
        else:
            second_activated_region="None"

    #   # 4️⃣ 파일 저장
        file_id = f"{file_name}_frame{frame_idx:04d}"
    #   cv2.imwrite(os.path.join(output_dir_box, f"{file_id}_roi.jpg"), frame_with_box)
    #   cv2.imwrite(os.path.join(output_dir_box, f"{file_id}_gradcam.jpg"), overlay_with_box)

    #  4️⃣ 영상 파일 저장
        video_writer_box.write(frame_with_box)
        video_writer_cam.write(overlay_with_box)

        first_detection_count[first_activated_region]+=1
        second_detection_count[second_activated_region]+=1
        for key in facial_region:
            if key!="None" and scores[key]!=-1:
                detection_probabillity[key]+=cam_score * scores[key]

        roi_result.append({
            'file_name': file_id,
            'cam_score': cam_score,
            'first_activate_region': first_activated_region,
            'second_activate_region': second_activated_region,
            'f_x1': f_x1, 'f_y1': f_y1, 'f_x2': f_x2, 'f_y2': f_y2,
            's_x1': s_x1, 's_y1': s_y1, 's_x2': s_x2, 's_y2': s_y2,
            **scores,
        })

        frame_idx += 1

    cap.release()
    video_writer_box.release()
    video_writer_cam.release()

    # Printing all the dictionaries
    first_detection_rate = {key: round((value / frame_idx)*100, 2) for key, value in first_detection_count.items()}
    second_detection_rate = {key: round((value / frame_idx)*100, 2) for key, value in second_detection_count.items()}

    # 📌 Note: A high proportion of 'None' may inflate the relative Contribution (%) and should be interpreted with caution.
    raw_detection_probabillity= {key: round(value, 4) for key, value in detection_probabillity.items()}
    probabillity_total = sum(detection_probabillity.values())
    detection_probabillity= {key: round((value/probabillity_total)*100, 2) for key, value in detection_probabillity.items()}

    # print("Video name:", file_name)
    # print("Facial Region:", facial_region)
    # print("First Detection Count:", first_detection_count)
    # print("Second Detection Count:", second_detection_count)
    # print("First Detection Rate:", first_detection_rate)
    # print("Second Detection Rate:", second_detection_rate)
    # print("Raw_Detection Probability:", raw_detection_probabillity)
    # print("Detection Probability:", detection_probabillity)

    roi_analyze_result = {
    "video_name": file_name,
    "binary_pred":result["Prediction"], 
    "cam_score":result["Probability"],
    "method_pred":result["Method"],
    "facial_region": facial_region,
    "first_detection_count": first_detection_count,
    "second_detection_count": second_detection_count,
    "first_detection_rate": first_detection_rate,
    "second_detection_rate": second_detection_rate,
    "raw_detection_probability": raw_detection_probabillity,
    "detection_probability": detection_probabillity
    }

    # 테이블 만들기
    table_data = []
    for region in facial_region:
        row = {
            "region": region,
            "first_count": f"{first_detection_count.get(region, '0')} ({first_detection_rate.get(region, '0.00')}%)",
            "second_count": f"{second_detection_count.get(region, '0')} ({second_detection_rate.get(region, '0.00')}%)",
            "confidence": "-" if detection_probabillity.get(region) is None else f"{detection_probabillity.get(region, '0.00')}%"
        }
        table_data.append(row)

    # 마지막 총합 행 추가
    table_data.append({
        "region": "total",
        "first_count": sum(int(v) for v in first_detection_count.values()),
        "second_count": sum(int(v) for v in second_detection_count.values()),
        "confidence": "100.00%"
    })


    return roi_analyze_result,table_data



def all_calculate_roi_scores(video_path,file_name,result,checkpoint_name='checkpoint_v35',selected_model='EfficientNet-b0'):
    print("여기요1")
    # 모델 구조를 정의
    model = Model(num_binary_classes=2, num_method_classes=7, model_name=selected_model).to(device)
    model.load_state_dict(torch.load(f'{checkpoint_path}/{checkpoint_name}.pt', map_location=device))
    model.eval()

    roi_analyze_result,table_data=analyze_roi_activation(video_path,file_name,result, model)

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
        ], check=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
        ], check=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print(f"✅ Video converted: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during conversion: {e}")


#여기부터
    def format_prompt(summary):
        print(f"summary: {summary}")
        prompt = (
            f" 아래에 제공된 데이터를 기반으로 모델이 해당 영상을 REAL 또는 FAKE로 판단한 근거를 쉽게 이해할 수 있도록 설명해 주세요."
            f"응답은 분석 내용만 포함하고, 다음 형식처럼 번호를 붙여주세요:\n"
            f"1. ...\n2. ...\n3. ...\n"
            f"모델 예측 결과:이 모델은 REAL/FAKE 판단에서 {summary['binary_pred']}로 예측했으며, 확률은 {summary['cam_score']}입니다. 딥페이크 기법 분류 결과: {summary['method_pred']}"
            f"참고 : original은 위조 흔적이 없는 원본 영상, others는 FaceForensics++의 5가지 기법 외의 위조 방식입니다."
            f"아래는 영상 {summary['video_name']}에 대한 Grad-CAM 기반 ROI 활성도 분석 요약입니다."

            f"분석 데이터 설명 (모든 값은 영상 전체 프레임을 통합한 통계 기반입니다):\n"
            f" - 가짜에 영향을 주는 부분을 grad-cam으로 시각화하여, 영상 내에 모든 프레임을 통합한 값입니다.\n"
            f" 1. [Facial Regions 분석 대상]: 분석에 사용된 얼굴 부위는 다음과 같습니다: {', '.join(summary['facial_region'])}. ('None'은 REAL로 판단되어 Grad-CAM이 그려지지 않은 경우입니다.)\n\n"
            f" 2. [1순위 활성화 횟수 (First Detection Count)]: 각 부위가 Grad-CAM에서 1순위로 가장 활성화된 프레임 수입니다.\n"
            f" {summary['first_detection_count']}\n\n"
            f" 3. [2순위 활성화 횟수 (Second Detection Count)]: 각 부위가 2순위로 활성화된 프레임 수입니다.\n"
            f" {summary['second_detection_count']}\n\n"
            f" 4. [1순위 활성화 비율 (First Detection Rate)]: 전체 프레임 중 각 부위가 1순위로 선택된 비율입니다 (% 단위).\n"
            f" {summary['first_detection_rate']}\n\n"
            f" 5. [2순위 활성화 비율 (Second Detection Rate)]: 전체 프레임 중 각 부위가 2순위로 선택된 비율입니다 (% 단위).\n"
            f" {summary['second_detection_rate']}\n\n"
            
        )



        # FAKE 판단에 대한 분석
        if summary['binary_pred'] == 'FAKE':
            prompt += (
                f" 6. [Raw Detection Probability]: 각 부위에 대해 Grad-CAM의 총 활성 기여도를 raw 점수로 나타낸 값입니다. 각 프레임별에서 (Fake일 확률 × 부위별 ROI 평균 활성도)를 계산하여, 부위별로 모두 더하여 구합니다. \n"
                f" {summary['raw_detection_probability']}\n\n"
                f" 7. [Normalized Detection Contribution (%)]: 위의 raw 값들을 전체 합으로 나눠 정규화한 값입니다. 각 부위가 전체 판단에 얼마나 기여했는지 상대적으로 보여줍니다. \n"
                f" {summary['detection_probability']}\n\n"
            )
            prompt += (
                "모델 결과가 FAKE로 판단된 경우, 딥페이크 기법 분류 결과와 위의 정보를 참고하여 어떤 얼굴 부위가 어떻게 작용했는지를 중심으로 분석하고, "
                "중요도가 높은 분석 내용을 선별하여 독자가 납득할 수 있도록 구체적으로 서술해 주세요.\n"
                "모델 결과가 FAKE로 판단된 경우에는 Grad-CAM 분석이 매우 중요하며, 특정 얼굴 부위들이 어떻게 활성화되었는지를 분석해 주세요.\n"
        )

        # REAL 판단에 대한 분석
        elif summary['binary_pred'] == 'REAL':
            prompt += (
            "모델 결과가 REAL로 판단된 경우, None의 detection_count 값을 참고하여 분석하고, 중요도가 높은 분석 내용을 선별하여 독자가 납득할 수 있도록 구체적으로 서술해 주세요.\n"
            "REAL로 판단된 경우, 얼굴 부위별 감지 횟수가 적고 'None'의 횟수가 대부분이기 때문에 REAL로 판단되었습니다. "
            "Grad-CAM이 그려지지 않거나, 활성화가 적어, 모델이 실제 영상이라고 예측한 이유를 명확히 할 수 있습니다. \n"
        )


        return prompt

    def query_model(prompt):
        url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "llama3",
            "prompt": prompt
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        print("응답 코드 : ",response.status_code)
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

    print("프롬프트 내용 : ",format_prompt(roi_analyze_result))
    response_txt=query_model(format_prompt(roi_analyze_result))
    return response_txt, table_data

