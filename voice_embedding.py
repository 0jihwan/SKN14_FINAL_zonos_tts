# 샘플 보이스를 줘서, 해당 발화자 목소리를 임베딩하는 코드

from zonos.model import Zonos
from zonos.utils import DEFAULT_DEVICE
from zonos.conditioning import make_cond_dict
import torchaudio
import os
import torch

dll_dir = r"C:\Program Files\eSpeak NG"
os.environ["PHONEMIZER_ESPEAK_PATH"] = os.path.join(dll_dir, "espeak-ng.exe")
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = os.path.join(dll_dir, "libespeak-ng.dll")


# 사전학습된 TTS 모델 로드
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=DEFAULT_DEVICE)



from pydub import AudioSegment

# m4a 형변환. 필요하면 사용、 필요없으면 주석처리
audio = AudioSegment.from_file("sample_file/joowoojae.m4a", format="m4a")
audio.export("sample_file/joowoojae.wav", format="wav")

# 원본 파일 업로드
wav, sr = torchaudio.load("sample_file/joowoojae.wav")
speaker_emb = model.make_speaker_embedding(wav, sr)
# 강제 형변환 <-날려도 되는지 체크
speaker_emb = speaker_emb.mean(dim=0, keepdim=True).unsqueeze(0)   # (1, 1, 256)



# 화자 저장
torch.save(speaker_emb.cpu(), "persona_list/joowoojae.pt")
print("화자 저장 완료")

# cond = make_cond_dict(
#     text="또 막 스커트 코디라고 해서. 핑크핑크한 그런 스커트 입을 필요 전혀 없어요. 블랙 원피스도 어딘가 살짝 컬러색이 들어간 걸 고르시거나. 아니면 넥라인이 살짝 파인 걸로 골라서요. 주어리로 얼굴 뒤에 밝은 포인트를 꼭 넣어 주는 거지.",
#     speaker=speaker_emb,
#     language="ko"
# )
#
# # espeak 강제 batch=1
# if isinstance(cond["espeak"], tuple):
#     t, l = cond["espeak"]
#     cond["espeak"] = ([t[0]], [l[0]])
#
# prefix = model.prepare_conditioning(cond)
#
# # 이거 False하면 C++ 컴파일 돌림 없으면 True하쇼
# codes = model.generate(prefix, disable_torch_compile=True)
# wavs  = model.autoencoder.decode(codes)
#
# # 누가 텐서 에러 띄우는지 범인 찾는 코드
# # for k, v in cond.items():
# #     if isinstance(v, torch.Tensor):
# #         print(k, v.shape)
#
# print("wavs.shape =", wavs.shape)
#
# # [B, C, T]?  -> [C, T]
# if wavs.ndim == 3 and wavs.shape[0] == 1:
#     wav = wavs.squeeze(0)   # (1, T)
# elif wavs.ndim == 2:
#     wav = wavs
# else:
#     raise RuntimeError(f"Unexpected wav shape {wavs.shape}")
#
#
# print("final wav.shape =", wav.shape)  # 확인
#
# torchaudio.save("output.wav", wav.cpu(), 44100)   # Zonos는 44kHz에 적합.