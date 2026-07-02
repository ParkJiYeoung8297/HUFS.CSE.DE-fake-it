import os
import uuid
import subprocess
import threading
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .services.uploads import save_uploaded_file
from .services.preprocessing import run_preprocessing
from .services.inference import run_inference
from .services.explainability import run_gradcam
from .services.llm import run_llm, warm_up_llm
from .utils.results import save_detection_result
from .detector.model_cache import preload_cached_models


preload_cached_models()
if os.environ.get("DEFAKE_LLM_WARMUP", "0") == "1":
    threading.Thread(target=warm_up_llm, daemon=True).start()


@csrf_exempt
def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        uploaded_file = request.FILES['video']

        try:
            filename, save_path = save_uploaded_file(uploaded_file)

            output_path = os.path.join(settings.MEDIA_ROOT, f"preprocessed_{filename}")

            preprocessed_path = run_preprocessing(save_path, output_path, uploaded_file.name)

            result = run_inference(preprocessed_path)

            response_txt = ""
            table_data = []

            if result["Prediction"] != "Unknown":
                roi_analyze_result, table_data = run_gradcam(
                    output_path,
                    uploaded_file.name,
                    result,
                )
                response_txt = run_llm(roi_analyze_result)

            response_data = {
                "message": "Success",
                "uploaded_video_name": uploaded_file.name,
                "saved_video_name": filename,
                "prediction": result["Prediction"],
                "probability": result["Probability"],
                "grad_cam_video_url": f"/media/preprocessed_{filename}/converted_grad_cam_on_original.mp4",
                "output_box_video_url": f"/media/preprocessed_{filename}/converted_output_box_on_original.mp4",
                "explanations": response_txt,
                "table_data": table_data,
            }
            result_json_path, result_jsonl_path = save_detection_result(
                response_data,
                output_path,
            )
            response_data["result_file"] = result_json_path
            response_data["result_log_file"] = result_jsonl_path

            # 저장 완료 후 파일 URL 반환
            return JsonResponse(response_data)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request"}, status=400)




@csrf_exempt
def show_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        uploaded_file = request.FILES['video']
        print(f"📥 Received file: {uploaded_file.name}")

        # 1. 원본 영상 임시 저장
        input_filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
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
            ], check=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
