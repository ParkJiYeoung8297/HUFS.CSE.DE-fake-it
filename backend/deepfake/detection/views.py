from django.shortcuts import render
import os
import uuid
import subprocess
import time
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .services.uploads import save_uploaded_file
from .services.preprocessing import run_preprocessing
from .services.inference import run_inference
from .services.explainability import run_gradcam
from .services.llm import run_llm
from .utils.performance import PerformanceLogger
from .df_model.model_cache import preload_cached_models


preload_cached_models()


def measure_elapsed(label, timings, func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    timings[label] = time.perf_counter() - start_time
    return result


def print_timing_summary(timings):
    print("\n===== Processing Time Summary =====")
    print(f"Average preprocessing time/video: {timings.get('preprocessing', 0.0):.2f} sec")
    print(f"Average inference time/video: {timings.get('inference', 0.0):.2f} sec")
    print(f"Grad-CAM generation time: {timings.get('grad_cam', 0.0):.2f} sec")
    print(f"LLM explanation generation time: {timings.get('llm', 0.0):.2f} sec")
    print(f"Total processing time/video: {timings.get('total', 0.0):.2f} sec")
    print("===================================\n")


@csrf_exempt
def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        total_start_time = time.perf_counter()
        timings = {}

        uploaded_file = request.FILES['video']
        performance_logger = PerformanceLogger(
            uploaded_file.name,
            getattr(uploaded_file, "size", None)
        )

        try:
            with performance_logger.step("total"):
                with performance_logger.step("file_save"):
                    filename, save_path = save_uploaded_file(uploaded_file)
                    performance_logger.set_video_path(save_path)

                output_path = os.path.join(settings.MEDIA_ROOT,f"preprocessed_{filename}")

                with performance_logger.step("preprocessing"):
                    preprocessed_path = run_preprocessing(save_path, output_path,uploaded_file.name,timings)

                with performance_logger.step("inference"):
                    result = run_inference(preprocessed_path,timings)

                response_txt = ""
                table_data = []

                if result["Prediction"] != "Unknown":
                    with performance_logger.step("grad_cam"):
                        roi_analyze_result, table_data = run_gradcam(
                            output_path,
                            uploaded_file.name,
                            result,
                            timings
                        )

                    with performance_logger.step("llm"):
                        response_txt = run_llm(roi_analyze_result, timings)

            timings['total'] = time.perf_counter() - total_start_time

            print_timing_summary(timings)
            performance_logger.save()

            # 저장 완료 후 파일 URL 반환
            return JsonResponse({
                "message": "Success",
                "prediction": result["Prediction"],
                "probability": result["Probability"],
                "grad_cam_video_url": f"/media/preprocessed_{filename}/converted_grad_cam_on_original.mp4",
                "output_box_video_url": f"/media/preprocessed_{filename}/converted_output_box_on_original.mp4",
                "explanations": response_txt,
                "table_data":table_data
        })
        except Exception as e:
            timings['total'] = time.perf_counter() - total_start_time
            performance_logger.save()
            return JsonResponse({"error": str(e)}, status=500)
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
