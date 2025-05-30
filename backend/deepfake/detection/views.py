from django.shortcuts import render
import os
import uuid
import subprocess
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        uploaded_file = request.FILES['video']
        print(f"📥 Received file: {uploaded_file.name}")

        # 고유한 파일명 생성
        filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
        save_path = os.path.join(settings.MEDIA_ROOT, filename)

        # 파일 저장
        with open(save_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        #         # 2. ✅ 전처리 수행
        # try:
        #     preprocessed_path = run_preprocessing(input_path)  # 전처리 함수 호출
        # except Exception as e:
        #     return JsonResponse({"error": "Preprocessing failed", "detail": str(e)}, status=500)

        # # 3. 결과 반환 (전처리된 파일 URL)
        # return JsonResponse({
        #     "message": "Video uploaded and processed successfully",
        #     "video_url": f"/media/{os.path.basename(preprocessed_path)}"
        # })

        # 저장 완료 후 파일 URL 반환
        return JsonResponse({
            "message": "Video uploaded successfully",
            "video_url": f"/media/{filename}"
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
