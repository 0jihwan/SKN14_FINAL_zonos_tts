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





---
초기 모델 tts 생성 시 약 40초 소요(3문장)



## Zonos 모델 runpod에서 사용법

1. runpod 로그인
2. 좌측 사이드바에서 Pods 선택
3. GPU 선택 (ex. RTX5090으로 진행)
4. start jupyter notebook 체크 후 Deploy on demand
5. jupyter notebook 열어서 접속 (접속되는데 시간이 조금 걸립니다)
6. 준비된 ipynb파일 업로드. 패키지 모두 설치
- Zonos 안에 있는 파일들 모두 Workspace로 옮기기(상위 폴더로 옮기기)
7. (선택) persona_list나 sample_file 같은 디렉토리를 만들거나, 코드를 수정
- 임베딩 파일도 업로드하거나 직접 돌려야합니다.