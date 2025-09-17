from zonos.model import Zonos
from zonos.utils import DEFAULT_DEVICE
import torch
import torchaudio
import datetime
import time
import os
from zonos.conditioning import make_cond_dict

dll_dir = r"C:\Program Files\eSpeak NG"
os.environ["PHONEMIZER_ESPEAK_PATH"] = os.path.join(dll_dir, "espeak-ng.exe")
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = os.path.join(dll_dir, "libespeak-ng.dll")

# --- 단계별 타이머 ---
def log_time(msg, start):
    now = time.time()
    print(f"{msg} 완료 ⏱ {(now - start):.2f} 초")
    return now

# 모델 허깅페이스에서 로드하는 코드
# print(">>> Loading Zonos model... (first time may take a few minutes)")
# t0 = time.time()
# model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=DEFAULT_DEVICE)
# t0 = log_time("모델 로딩", t0)
# print(">>> Model loaded on", model.device)

# 모델 애초에 로컬(도커이미지)에 저장해놓고 쓰는 코드
print(">>> Loading Zonos Hybrid model from local...")
model = Zonos.from_local(
    config_path="Zonos-v0.1-transformer/config.json",
    model_path="Zonos-v0.1-transformer/model.safetensors",
    device=DEFAULT_DEVICE
)
print(">>> Hybrid model loaded on", model.device)


# --- 임베딩 불러오기 ---
DEFAULT_SPEAKER = "joowoojae"
t1 = time.time()
speaker_emb = torch.load(f"persona_list/{DEFAULT_SPEAKER}.pt").to(DEFAULT_DEVICE)
t1 = log_time("임베딩 불러오기", t1)

text = "안녕하시렵니까? 오늘의 주우재의 주우재입니다. 오늘도 주우재의 살끼요 말까요?"
#  진행해보도록 하겠습니다. 오 알파인더스트리의 마원이네요. 확실히 패션이 돌고 돈다고 생각되는게 한 3,4년 전에 마원이 유행했었잖아요.
cond = make_cond_dict(text=text, speaker=speaker_emb, language="ko")
if isinstance(cond["espeak"], tuple):  # espeak 강제 batch=1
    t, l = cond["espeak"]
    cond["espeak"] = ([t[0]], [l[0]])
t1 = log_time("conditioning 준비", t1)

def estimate_tokens(text: str) -> int:
    return max(100, len(text) * 20)

start_time = time.time()
with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    prefix = model.prepare_conditioning(cond)
    t1 = log_time("prefix 준비", start_time)

    max_new_tokens = max(64, len(text) * 4)
    codes = model.generate(prefix, disable_torch_compile=True, progress_bar=False, max_new_tokens=max_tokens)
    t1 = log_time("코드 생성", t1)

    wavs = model.autoencoder.decode(codes)
    t1 = log_time("오디오 디코딩", t1)

# --- wav 정리 ---
if wavs.ndim == 3 and wavs.shape[0] == 1:
    wav = wavs.squeeze(0)
elif wavs.ndim == 2:
    wav = wavs
else:
    raise RuntimeError(f"Unexpected wav shape {wavs.shape}")
t1 = log_time("wav 정리", t1)

# --- 무음 제거 ---
wav_np = wav.cpu().numpy()
wav_trimmed, _ = librosa.effects.trim(wav_np, top_db=30)
wav_tensor = torch.tensor(wav_trimmed).unsqueeze(0)  # [1, T]


# --- 저장 ---
now = datetime.datetime.now()
out_path = f"output_{now.strftime('%m%d_%H%M')}.wav"
torchaudio.save(out_path, wav.cpu(), 44100)
t1 = log_time("파일 저장", t1)

# --- 총 소요시간 ---
end_time = time.time()
print("final wav.shape =", wav.shape)
print(f"총 소요시간: {(end_time - start_time):.2f} 초")
print(f"저장된 파일: {out_path}")
