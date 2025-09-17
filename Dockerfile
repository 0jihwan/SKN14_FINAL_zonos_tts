FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg git git-lfs espeak-ng && rm -rf /var/lib/apt/lists/*

# 나중에 모델 돌아가는거 확인 했을때, 이 주석 해제
# RUN git lfs install
# RUN git clone https://huggingface.co/Zyphra/Zonos-v0.1-transformer /app/Zonos-v0.1-transformer


WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "/app/handler.py"]