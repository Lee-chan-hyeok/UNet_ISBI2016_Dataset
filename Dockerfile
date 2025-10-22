FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYTHONIOENCODING=UTF-8

# 필수 패키지 설치 (Python 3.9 사용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3.9-dev python3.9-distutils python3-pip \
    git vim curl wget \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Python 3.9을 기본으로 설정
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --set python3 /usr/bin/python3.9

# pip 최신화 및 빌드에 필요한 패키지 설치
RUN python3 -m pip install --upgrade pip setuptools wheel

# CPU 버전 PyTorch 설치 (UNet을 PyTorch로 구현하는 경우)
RUN pip install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 작업 디렉토리 설정
WORKDIR /workspace

# requirements.txt 복사 및 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 기본 명령어
CMD ["python3"]