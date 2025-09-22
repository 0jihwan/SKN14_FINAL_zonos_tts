from zonos.model import Zonos
from zonos.utils import DEFAULT_DEVICE
import torch
import torchaudio
import datetime
import time
import os
import librosa
import re
import unicodedata
from zonos.conditioning import make_cond_dict

dll_dir = r"C:\Program Files\eSpeak NG"
os.environ["PHONEMIZER_ESPEAK_PATH"] = os.path.join(dll_dir, "espeak-ng.exe")
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = os.path.join(dll_dir, "libespeak-ng.dll")

# --- 단계별 타이머 ---
def log_time(msg, start):
    now = time.time()
    print(f"{msg} 완료 ⏱ {(now - start):.2f} 초")
    return now

# --- 텍스트 정규화 ---
def clean_ko_text(s: str) -> str:
    s = re.sub(r'[\u200B-\u200D\uFEFF\u2060]', '', s)
    s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def ensure_terminal_punct(s: str) -> str:
    return s if re.search(r'[.?!…]\s*$', s) else s + '.'

# --- 문장 단위 split 함수 ---
def split_sentences(text: str, max_len=80):
    sentences = re.split(r'(?<=[.?!])', text)
    results = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) > max_len:
            subs = re.split(r'(?<=,)', s)
            results.extend([ensure_terminal_punct(sub.strip()) for sub in subs if sub.strip()])
        else:
            results.append(ensure_terminal_punct(s))
    return results

# --- max_new_tokens 계산 ---
def estimate_tokens_ko(s: str) -> int:
    chars_no_space = len(re.sub(r'\s+', '', s))
    return max(1024, min(chars_no_space * 72, 16384))

# --- wav shape 강제 ---
def to_CT(wav: torch.Tensor) -> torch.Tensor:
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    elif wav.ndim != 2:
        raise RuntimeError(f"Unexpected wav shape {wav.shape}")
    return wav

# --- 말미 페이드아웃 ---
def apply_fade_out(wav: torch.Tensor, sr: int, fade_ms: float = 120.0) -> torch.Tensor:
    assert wav.ndim == 2
    T = wav.shape[-1]
    n = max(1, min(int(sr * (fade_ms/1000.0)), T-1))
    ramp = torch.linspace(1.0, 0.0, n, dtype=wav.dtype, device=wav.device)
    wav[..., -n:] = wav[..., -n:] * ramp
    return wav


# --- 모델 로드 ---
print(">>> Loading Zonos Transformer model from local...")
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=DEFAULT_DEVICE)
print(">>> Model loaded on", model.device)

# --- 임베딩 불러오기 ---
DEFAULT_SPEAKER = "joowoojae"
t1 = time.time()
speaker_emb = torch.load(f"persona_list/{DEFAULT_SPEAKER}.pt").to(DEFAULT_DEVICE)
t1 = log_time("임베딩 불러오기", t1)

# --- 테스트 텍스트 ---
text = "재활용 폴리에스터는 플라스틱 병과 폐기물을 재활용해 환경오염과 에너지 소모를 줄이며, 친환경적이고 지속 가능하다"
text = clean_ko_text(text)

start_time = time.time()
final_wavs = []

# --- 문장별 분리하여 처리 ---
# for sent in split_sentences(text):
#     cond = make_cond_dict(text=sent, speaker=speaker_emb, language="ko")
#     if isinstance(cond["espeak"], tuple):
#         t, l = cond["espeak"]
#         cond["espeak"] = ([t[0]], [l[0]])

#     with torch.inference_mode():
#         prefix = model.prepare_conditioning(cond)
#         max_tokens = estimate_tokens_ko(sent)

#         codes = model.generate(prefix, disable_torch_compile=True, progress_bar=False, max_new_tokens=max_tokens)
#         wavs = model.autoencoder.decode(codes)

#     if wavs.ndim == 3 and wavs.shape[0] == 1:
#         wav = wavs.squeeze(0)
#     elif wavs.ndim == 2:
#         wav = wavs
#     else:
#         raise RuntimeError(f"Unexpected wav shape {wavs.shape}")

#     final_wavs.append(wav)

# # --- 문장별 wav 이어붙이기 ---
# wav_full = torch.cat(final_wavs, dim=-1)

# --- 텍스트 전체를 한 번에 처리 ---
cond = make_cond_dict(text=text, speaker=speaker_emb, language="ko")
if isinstance(cond["espeak"], tuple):
    t, l = cond["espeak"]
    cond["espeak"] = ([t[0]], [l[0]])

with torch.inference_mode():
    prefix = model.prepare_conditioning(cond)
    max_tokens = estimate_tokens_ko(text)

    codes = model.generate(prefix, disable_torch_compile=True, progress_bar=False, max_new_tokens=max_tokens)
    wavs = model.autoencoder.decode(codes)

if wavs.ndim == 3 and wavs.shape[0] == 1:
    wav_full = wavs.squeeze(0)
elif wavs.ndim == 2:
    wav_full = wavs
else:
    raise RuntimeError(f"Unexpected wav shape {wavs.shape}")

t1 = log_time("텍스트 전체 처리 완료", start_time)


# --- 앞부분만 트림 (뒤는 보존) ---
wav_np = wav_full.cpu().numpy()
_, idx = librosa.effects.trim(wav_np, top_db=18)
start_idx = int(idx[0])
head_trimmed = wav_np[..., start_idx:]

# --- Tensor 변환 & 보정 ---
wav_tensor = torch.tensor(head_trimmed)
wav_tensor = to_CT(wav_tensor)

# --- 말미 무음 패딩 + 페이드아웃 ---
sr = 44100
pad_end_ms = 500
pad = int(sr * (pad_end_ms/1000.0))
sil = torch.zeros((wav_tensor.shape[0], pad), dtype=wav_tensor.dtype, device=wav_tensor.device)
wav_tensor = torch.cat([wav_tensor, sil], dim=-1)

wav_tensor = apply_fade_out(wav_tensor, sr=sr, fade_ms=120.0)

t1 = log_time("트림/패딩/페이드", t1)

# --- 저장 ---
now = datetime.datetime.now()
out_path = f"output_{now.strftime('%m%d_%H%M')}.wav"
torchaudio.save(out_path, wav_tensor.cpu(), sr)
t1 = log_time("파일 저장", t1)

# --- 총 소요시간 ---
end_time = time.time()
print("final wav.shape =", wav_tensor.shape)
print(f"총 소요시간: {(end_time - start_time):.2f} 초")
print(f"저장된 파일: {out_path}")
