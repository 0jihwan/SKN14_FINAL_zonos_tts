FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg git git-lfs espeak-ng && rm -rf /var/lib/apt/lists/*

# 나중에 모델 돌아가는거 확인 했을때, 이 주석 해제
# RUN git lfs install
# RUN git clone https://huggingface.co/Zyphra/Zonos-v0.1-transformer /app/Zonos-v0.1-transformer

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 로컬에서 Zonos 모델 파일 복사
RUN mkdir -p /app/models
RUN git lfs install & git clone https://huggingface.co/Zyphra/Zonos-v0.1-transformer /app/models/Zonos-v0.1-transformer

COPY . .

# 절대경로로 실행 (WORKDIR 무시 케이스 방지)
CMD ["python", "/app/handler.py"]
