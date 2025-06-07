from django.shortcuts import render
import os
import uuid
import subprocess
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .df_model.preprocessing import process_single_video
from .df_model.detection_model import run_detection_model
# from .df_model.grad_cam import calculate_roi_scores
from .df_model.all_grad_cam import all_calculate_roi_scores


@csrf_exempt
def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        uploaded_file = request.FILES['video']
        print(f"📥 Received file: {uploaded_file.name}")

        # 고유한 파일명 생성
        filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
        save_path = os.path.join(settings.MEDIA_ROOT, filename)
        output_path=os.path.join(settings.MEDIA_ROOT, f"preprocessed_{filename}")
        # 파일 저장
        with open(save_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # 1. ✅ 전처리 수행
        try:
            preprocessed_path = process_single_video(save_path,output_path,uploaded_file.name)  # 전처리 함수 호출
        except Exception as e:
            return JsonResponse({"error": "Preprocessing failed", "detail": str(e)}, status=500)

        # 2. ✅ 탐지 모델
        try:
            result =  run_detection_model(preprocessed_path)
        except Exception as e:
            return JsonResponse({"error": "Detection failed", "detail": str(e)}, status=500)
        # 3. ✅ Grad_cam
        final_result = None  # 초기화
        if result["Prediction"]=="FAKE":
            try:
                response_txt,grad_cam_path,output_dir_box=all_calculate_roi_scores(output_path,uploaded_file.name,checkpoint_name='checkpoint_1')

            except Exception as e:
                return JsonResponse({"error": "Grad_cam failed", "detail": str(e)}, status=500)

        # 저장 완료 후 파일 URL 반환
        return JsonResponse({
            "message": "Success",
            "prediction": result["Prediction"],
            "probability": result["Probability"],
            "grad_cam_video_url": f"/media/preprocessed_{filename}/converted_grad_cam_on_original.mp4",
            "output_box_video_url": f"/media/preprocessed_{filename}/converted_output_box_on_original.mp4",
            "explanations": response_txt
            # "final_result":final_result  #직렬화 필요함
        })

    return JsonResponse({"error": "Invalid request"}, status=400)



@csrf_exempt
def show_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        uploaded_file = request.FILES['video']
        print(f"📥 Received file: {uploaded_file.name}")

        # 1. 원본 영상 임시 저장
        input_filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
        input_path = os.path.join(settings.MEDIA_ROOT, input_filename)

        with open(input_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # 2. 변환된 영상 저장 경로
        output_filename = f"converted_{uuid.uuid4().hex}.mp4"
        output_path = os.path.join(settings.MEDIA_ROOT, output_filename)

        # 3. ffmpeg 변환 수행
        try:
            subprocess.run([
                'ffmpeg', '-i', input_path,
                '-vcodec', 'libx264',
                '-acodec', 'aac',
                '-strict', 'experimental',
                '-y',  # 덮어쓰기
                output_path
            ], check=True)

            print(f"✅ Video converted: {output_path}")

            # 4. 성공 응답
            return JsonResponse({
                "message": "Video uploaded and converted successfully",
                "video_url": f"/media/{output_filename}"
            })

        except subprocess.CalledProcessError as e:
            return JsonResponse({
                "error": "ffmpeg conversion failed",
                "detail": str(e)
            }, status=500)

        finally:
            # 원본은 삭제
            if os.path.exists(input_path):
                os.remove(input_path)

    return JsonResponse({"error": "Invalid request"}, status=400)
