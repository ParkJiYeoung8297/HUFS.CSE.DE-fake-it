FROM python:3.8-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 \
    cmake g++ build-essential
# 필수 패키지 설치
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

# Python 패키지 설치
RUN pip install --no-cache-dir \
    numpy \
    opencv-python \
    pandas \
    torch torchvision \
    face_recognition \
    tqdm matplotlib

# 작업 디렉토리 설정
WORKDIR /app

# 소스코드 복사
COPY preprocessing.py .

# 실행할 명령어
CMD ["python", "preprocessing.py"]
