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


<<<<<<< HEAD
---
### 최적화에 적용된 방법
1. dockerignore에 docker 배포 시 사용되지 않는 로컬 전용 파일 추가
2. huggingface에서 불러오는 모델을 docker에서 직접 로드(처음 배포시 상당한 시간 소요, 하지만 작업 속도는 상승)
3. 토큰 수 조절
=======



---
초기 모델 tts 생성 시 약 40초 소요(3문장)
최적화에 사용한 방법
>>>>>>> main
