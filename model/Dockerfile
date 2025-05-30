FROM python:3.8-slim

# 시스템 패키지 설치 (중복 제거)
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 \
    cmake g++ build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드 및 주요 패키지 설치
RUN pip install --no-cache-dir --upgrade pip

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 기본 실행 명령
CMD ["bash"]
