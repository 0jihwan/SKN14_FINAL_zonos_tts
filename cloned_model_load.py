from zonos.model import Zonos
from zonos.utils import DEFAULT_DEVICE
import torch
import torchaudio
import datetime
import time
import os
from voice_embedding import model

start_time = time.time()

dll_dir = r"C:\Program Files\eSpeak NG"
os.environ["PHONEMIZER_ESPEAK_PATH"] = os.path.join(dll_dir, "espeak-ng.exe")
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = os.path.join(dll_dir, "libespeak-ng.dll")


now = datetime.datetime.now()

# model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=DEFAULT_DEVICE)

# 저장된 임베딩 불러오기
speaker_emb = torch.load("persona_list/HongJinkyeong.pt").to(DEFAULT_DEVICE)

# text = "이 룩은 정말 간편하면서도 스타일리시해요. 어떤 상황에서도 편안함을 느낄 수 있는 조화로운 조합이거든요. 여러분은 이 룩을 어떻게 생각해요? 혹시 여러분의 스타일은 어떤 거예요?"
# text = "또 막 스커트 코디라고 해서. 핑크핑크한 그런 스커트 입을 필요 전혀 없어요. 블랙 원피스도 어딘가 살짝 컬러색이 들어간 걸 고르시거나. 아니면 넥라인이 살짝 파인 걸로 골라서요. 주어리로 얼굴 뒤에 밝은 포인트를 꼭 넣어 주는 거지."
# text = "원래 그검정색을 끼면은 대비가 좀 심하다고 해야하나? 꽉 그냥 쨍한 이런게 더 자연스럽더라고요. 그래서 뿔테 부담스러운 분들은 그레이 이런거로. 네 그럼 좀 덜 부담스러울 수도 있어요."
text = "반가워요."
# TTS 조건 생성
from zonos.conditioning import make_cond_dict
cond = make_cond_dict(
    text=text,
    speaker=speaker_emb,
    language="ko",

)
print("cond done.")
# espeak 강제 batch=1
if isinstance(cond["espeak"], tuple):
    t, l = cond["espeak"]
    cond["espeak"] = ([t[0]], [l[0]])

use_cuda = torch.cuda.is_available()
autocast_ctx = torch.cuda.amp.autocast if use_cuda else torch.autocast  # CPU면 자동 비활성

with torch.inference_mode():
    if use_cuda:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            prefix = model.prepare_conditioning(cond)
    else:
        # CPU일 땐 AMP가 큰 의미 없으니 그냥 실행
        prefix = model.prepare_conditioning(cond)
# prefix = model.prepare_conditioning(cond)
print("prefix done.")

# 이거 False하면 C++ 컴파일 돌림 없으면 True하쇼
codes = model.generate(prefix, disable_torch_compile=True, progress_bar=False)  # 얘가 너무 오래걸려 해결필요
print("codes done.")
wavs  = model.autoencoder.decode(codes)
print("wavs done.")

# [B, C, T]?  -> [C, T]
if wavs.ndim == 3 and wavs.shape[0] == 1:
    wav = wavs.squeeze(0)   # (1, T)
elif wavs.ndim == 2:
    wav = wavs

# if wavs.ndim == 3:  # [B, 1, T]
#     wavs = wavs.squeeze(1)   # -> [B, T]
#     wav = torch.cat([w for w in wavs], dim=-1)  # 이어붙임
# elif wavs.ndim == 2:  # [1, T] 한 문장만
#     wav = wavs

else:
    raise RuntimeError(f"Unexpected wav shape {wavs.shape}")


print("final wav.shape =", wav.shape)  # 확인
torchaudio.save(f"output_{now.strftime("%m%d_%H%M")}.wav", wav.cpu(), 44100)
end_time = time.time()
print(f"소요시간: {(end_time - start_time):.2f} 초")