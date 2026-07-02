import os
import subprocess
import threading
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .services.llm import warm_up_llm
from .services.pipeline import analyze_uploaded_video
from .services.uploads import convert_video, save_uploaded_file
from .detector.model_cache import preload_cached_models


preload_cached_models()
if os.environ.get("DEFAKE_LLM_WARMUP", "0") == "1":
    threading.Thread(target=warm_up_llm, daemon=True).start()


@csrf_exempt
def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        uploaded_file = request.FILES['video']

        try:
            return JsonResponse(analyze_uploaded_video(uploaded_file))
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request"}, status=400)




@csrf_exempt
def show_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        uploaded_file = request.FILES['video']
        print(f"📥 Received file: {uploaded_file.name}")

        filename, input_path = save_uploaded_file(uploaded_file)
        try:
            output_filename, _output_path = convert_video(input_path)

            return JsonResponse({
                "message": "Video uploaded and converted successfully",
                "saved_video_name": filename,
                "video_url": f"/media/{output_filename}"
            })

        except subprocess.CalledProcessError as e:
            return JsonResponse({
                "error": "ffmpeg conversion failed",
                "detail": str(e)
            }, status=500)

        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    return JsonResponse({"error": "Invalid request"}, status=400)
