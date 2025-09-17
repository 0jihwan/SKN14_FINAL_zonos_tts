# SKN14_FINAL_zonos_tts




### 구동 메뉴얼

공통)
- espeak-ng 설치 필요 (https://github.com/espeak-ng/espeak-ng/releases)
- espeak을 읽기 위해 환경변수 지정 필요

- 필요한 모듈(runpod 가상환경 포함)
- requirements.txt 참조



로컬)
- 윈도우) 여전히 오류 시 https://github.com/bootphon/phonemizer/issues/44 참조





가상환경)
apt-get update
apt-get install espeak-ng
apt-get install -y ffmpeg
pip install py-espeak-ng

pip install sudachipy sudachidict_core inflect phonemizer kanjize phonemizer sudachipy sudachidict_full safetensors huggingface_hub transformers
git clone https://github.com/Zyphra/Zonos.git


만약에 python을 못읽으면?settings.json에 아래 추가하면 됨.
```
    "files.encoding": "utf8",
    "files.autoGuessEncoding": false,
    "files.eol": "\n",
    "files.insertFinalNewline": true,
    "files.trimTrailingWhitespace": true
```