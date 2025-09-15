# espeak의 dll파일을 인식하지 못할 때, 추적/테스트하는 코드

import os
import ctypes

dll_dir = r"C:\Program Files\eSpeak NG"
os.environ["PHONEMIZER_ESPEAK_PATH"] = os.path.join(dll_dir, "espeak-ng.exe")
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = os.path.join(dll_dir, "libespeak-ng.dll")

# DLL 직접 로드 (혹시 몰라 강제)
# ctypes.CDLL(os.environ["PHONEMIZER_ESPEAK_LIBRARY"])
# os.add_dll_directory(dll_dir)

# 테스트
from phonemizer.backend import EspeakBackend
backend = EspeakBackend("en-us")
print(backend.phonemize(["hello world"], strip=True))
