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
model = Zonos.from_local(
    config_path="Zonos-v0.1-transformer/config.json",
    model_path="Zonos-v0.1-transformer/model.safetensors",
    device=DEFAULT_DEVICE
)
print(">>> Model loaded on", model.device)

# --- 임베딩 불러오기 ---
DEFAULT_SPEAKER = "joowoojae"
t1 = time.time()
speaker_emb = torch.load(f"persona_list/{DEFAULT_SPEAKER}.pt").to(DEFAULT_DEVICE)
t1 = log_time("임베딩 불러오기", t1)

# --- 테스트 텍스트 ---
text = "안녕하십니까? 오늘의 주우재의 주우재입니다. 오늘도 주우재의 살까요 말까요?"

start_time = time.time()
final_wavs = []

# --- 문장별 처리 ---
for sent in split_sentences(text):
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

    final_wavs.append(wav)

# --- 문장별 wav 이어붙이기 ---
wav_full = torch.cat(final_wavs, dim=-1)
t1 = log_time("문장 합치기", start_time)

# --- 무음 제거 ---
wav_np = wav_full.cpu().numpy()
wav_trimmed, _ = librosa.effects.trim(wav_np, top_db=30)
wav_tensor = torch.tensor(wav_trimmed)

if wav_tensor.ndim == 1:
    wav_tensor = wav_tensor.unsqueeze(0)  # [1, T]
    print("1D")
elif wav_tensor.ndim == 2:
    print("2D")  # 이미 [C, T]
else:
    raise RuntimeError(f"Unexpected trimmed wav shape {wav_tensor.shape}")
t1 = log_time("무음 제거", t1)

# --- 저장 ---
now = datetime.datetime.now()
out_path = f"output_{now.strftime('%m%d_%H%M')}.wav"
torchaudio.save(out_path, wav_tensor.cpu(), 44100)
t1 = log_time("파일 저장", t1)

# --- 총 소요시간 ---
end_time = time.time()
print("final wav.shape =", wav_tensor.shape)
print(f"총 소요시간: {(end_time - start_time):.2f} 초")
print(f"저장된 파일: {out_path}")
