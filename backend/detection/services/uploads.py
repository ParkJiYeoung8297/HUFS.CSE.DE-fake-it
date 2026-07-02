import os
import uuid
import subprocess

from django.conf import settings


def save_uploaded_file(uploaded_file):
    filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"

    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    save_path = os.path.join(settings.MEDIA_ROOT, filename)

    with open(save_path, 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)

    return filename, save_path

def convert_video(input_path):
    output_filename = f"converted_{uuid.uuid4().hex}.mp4"
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    output_path = os.path.join(settings.MEDIA_ROOT, output_filename)

    subprocess.run([
        'ffmpeg', '-i', input_path,
        '-vcodec', 'libx264',
        '-acodec', 'aac',
        '-strict', 'experimental',
        '-y',
        output_path
    ], check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)

    return output_filename, output_path
