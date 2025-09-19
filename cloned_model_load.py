from zonos.model import Zonos
from zonos.utils import DEFAULT_DEVICE
import torch
import torchaudio
import datetime
import time
import os
import librosa
import re
from zonos.conditioning import make_cond_dict
import uuid

# --- espeak DLL 경로 (Windows 로컬) ---
dll_dir = r"C:\Program Files\eSpeak NG"
os.environ["PHONEMIZER_ESPEAK_PATH"] = os.path.join(dll_dir, "espeak-ng.exe")
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = os.path.join(dll_dir, "libespeak-ng.dll")

# --- 단계별 타이머 ---
def log_time(msg, start):
    now = time.time()
    print(f"{msg} 완료 ⏱ {(now - start):.2f} 초")
    return now

# --- 문장 단위 split 함수 ---
def split_sentences(text: str):
    sentences = re.split(r'(?<=[.?!?,])', text)
    return [s.strip() for s in sentences if s.strip()]

# --- 모델 로드 ---
print(">>> Loading Zonos Transformer model from local...")
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=DEFAULT_DEVICE)
print(">>> Model loaded on", model.device)

# --- 임베딩 불러오기 ---
DEFAULT_SPEAKER = "hanhyaejin"
t1 = time.time()
speaker_emb = torch.load(f"persona_list/{DEFAULT_SPEAKER}.pt").to(DEFAULT_DEVICE)
t1 = log_time("임베딩 불러오기", t1)

# --- 테스트 텍스트 ---
text = "안녕하세요 한혜진이에요. 제가 ESG 패션에 대해 알려드릴게요. 오늘은 친환경 소재 이야기를 해보겠습니다."

# --- 세션 ID & 준비 ---
session_id = uuid.uuid4().hex[:6]   # 랜덤 6자리
sentences = split_sentences(text)
total = len(sentences)

start_time = time.time()
results = []

# --- 문장별 처리 ---
for idx, sent in enumerate(sentences, start=1):
    cond = make_cond_dict(text=sent, speaker=speaker_emb, language="ko")
    if isinstance(cond["espeak"], tuple):  # espeak 강제 batch=1
        t, l = cond["espeak"]
        cond["espeak"] = ([t[0]], [l[0]])

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        prefix = model.prepare_conditioning(cond)

        # 문장 길이에 따라 max_new_tokens 조정
        max_tokens = max(64, len(sent) * 24)
        codes = model.generate(prefix, disable_torch_compile=True, progress_bar=False, max_new_tokens=max_tokens)
        wavs = model.autoencoder.decode(codes)

    # wav 정리
    if wavs.ndim == 3 and wavs.shape[0] == 1:
        wav = wavs.squeeze(0)
    elif wavs.ndim == 2:
        wav = wavs
    else:
        raise RuntimeError(f"Unexpected wav shape {wavs.shape}")

    # 무음 제거
    wav_np = wav.cpu().numpy()
    wav_trimmed, _ = librosa.effects.trim(wav_np, top_db=10)
    wav_tensor = torch.tensor(wav_trimmed, dtype=torch.float32)

    if wav_tensor.ndim == 1:
        wav_tensor = wav_tensor.unsqueeze(0)  # [1, T]

    # --- 파일 저장 (문장별) ---
    now = datetime.datetime.now()
    out_path = f"output_{DEFAULT_SPEAKER}_{session_id}_{total}-{idx}.wav"
    torchaudio.save(out_path, wav_tensor.cpu(), 44100)

    print(f"저장 완료: {out_path} (shape={wav_tensor.shape})")
    results.append(out_path)

t1 = log_time("문장별 파일 저장", start_time)

# --- 총 소요시간 ---
end_time = time.time()
print("총 문장 수:", total)
print("총 소요시간: %.2f 초" % (end_time - start_time))
print("저장된 파일 목록:", results)
