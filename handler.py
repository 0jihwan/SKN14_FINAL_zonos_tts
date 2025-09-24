import os, time, datetime, boto3, torch, torchaudio
import librosa, re, unicodedata
from runpod import serverless
from zonos.utils import DEFAULT_DEVICE
from zonos.conditioning import make_cond_dict
from zonos.model import Zonos

# espeak 변수
os.environ["PHONEMIZER_ESPEAK_PATH"] = "/usr/bin/espeak-ng"
os.environ["ESPEAK_DATA_PATH"] = "/usr/share/espeak-ng-data"

# AWS 환경 변수
S3_BUCKET = os.getenv("AWS_S3_BUCKET")
REGION = os.getenv("AWS_REGION", "ap-northeast-2")
PREFIX = os.getenv("S3_FOLDER_PREFIX", "tts")

# --- 모델 로딩 (HuggingFace에서만 불러오기) ---
print(">>> CWD:", os.getcwd())
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=DEFAULT_DEVICE)

s3 = boto3.client("s3", region_name=REGION)

# 화자 딕셔너리 매핑
SPEAKER_MAP = {
    "1": "HongJinkyeong",
    "2": "joowoojae",
    "3": "hanhyaejin",
    "4": "kimhoyeong"
}

# 기본 persona
DEFAULT_SPEAKER_ID = "1"
DEFAULT_SPEAKER = SPEAKER_MAP[DEFAULT_SPEAKER_ID]
DEFAULT_PATH = f"persona_list/{DEFAULT_SPEAKER}.pt"
default_emb = torch.load(DEFAULT_PATH).to(DEFAULT_DEVICE)

# --- 텍스트 전처리 함수 ---
def clean_ko_text(s: str) -> str:
    # 제로폭 문자 제거 + NFKC 정규화 + 공백 정리
    s = re.sub(r'[\u200B-\u200D\uFEFF\u2060]', '', s)
    s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def ensure_terminal_punct(s: str) -> str:
    # 문장 끝 종결부호 보장 (없으면 마침표 추가)
    return s if re.search(r'[.?!…]\s*$', s) else s + '.'

# --- 문장 단위 split 함수 ---
def split_sentences(text: str, max_len=80):
    # .?! 로만 자름 (, 제거)
    sentences = re.split(r'(?<=[.?!])', text)
    results = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) > max_len:  # 너무 긴 경우 comma 기준 분리
            subs = re.split(r'(?<=,)', s)
            results.extend([ensure_terminal_punct(sub.strip()) for sub in subs if sub.strip()])
        else:
            results.append(ensure_terminal_punct(s))
    return results

def estimate_tokens_ko(s: str) -> int:
    # 한국어 안전 계수 (보수적): 글자당 64~80 토큰, 최소 512
    chars_no_space = len(re.sub(r'\s+', '', s))
    return max(1024, min(chars_no_space * 72, 16384))

# --- wav shape 보정 ---
def to_CT(wav: torch.Tensor) -> torch.Tensor:
    # librosa에서 넘어오면 1D(T,)일 수 있음 → [1, T]로 통일
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    elif wav.ndim != 2:
        raise RuntimeError(f"Unexpected wav shape {wav.shape} (want [C, T])")
    return wav

# --- 말미 페이드아웃 ---
def apply_fade_out(wav: torch.Tensor, sr: int, fade_ms: float = 120.0) -> torch.Tensor:
    assert wav.ndim == 2
    T = wav.shape[-1]
    n = max(1, min(int(sr * (fade_ms / 1000.0)), T - 1))
    ramp = torch.linspace(1.0, 0.0, n, dtype=wav.dtype, device=wav.device)
    wav[..., -n:] = wav[..., -n:] * ramp
    return wav


def handler(job):
    try:
        start_time = time.time()

        text = job["input"].get("text", "안녕하세요")
        persona_input = job["input"].get("persona", DEFAULT_SPEAKER)

        # persona_input이 숫자인 경우 이름으로 변환
        # 숫자가 아니거나 매핑에 없는 경우 기본값으로 설정
        if str(persona_input) in SPEAKER_MAP:
            persona_name = SPEAKER_MAP[str(persona_input)]
        else:
            persona_name = DEFAULT_SPEAKER

        # persona embedding 불러오기
        speaker_path = f"persona_list/{persona_name}.pt"
        emb = torch.load(speaker_path).to(DEFAULT_DEVICE) if os.path.exists(speaker_path) else default_emb

        final_wavs = []

        # 텍스트 정리
        text = clean_ko_text(text)

        # 문장 단위로 나눠서 처리
        for sent in split_sentences(text):
            # conditioning
            cond = make_cond_dict(text=sent, speaker=emb, language="ko")
            if isinstance(cond["espeak"], tuple):  # espeak 강제 batch=1
                t, l = cond["espeak"]
                cond["espeak"] = ([t[0]], [l[0]])

            # prefix 준비
            with torch.inference_mode():
                prefix = model.prepare_conditioning(cond)

                # --- 문장 길이에 따른 max_new_tokens 계산 (보수적)
                max_tokens = estimate_tokens_ko(sent)

                # 코드 생성 & 오디오 복원
                codes = model.generate(
                    prefix,
                    disable_torch_compile=True,
                    progress_bar=False,
                    max_new_tokens=max_tokens
                )
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

        # --- 앞부분만 무음 제거 (뒤는 보존) ---
        wav_np_full = wav_full.cpu().numpy()
        _, idx = librosa.effects.trim(wav_np_full, top_db=18)  # 덜 공격적
        start_idx = int(idx[0])                                # 앞만 잘라냄
        head_trimmed = wav_np_full[..., start_idx:]

        # --- 텐서화 + [C, T] 강제 ---
        wav_tensor = torch.tensor(head_trimmed)
        wav_tensor = to_CT(wav_tensor)

        # --- 말미 여유 무음 패딩 (0.5초) ---
        sr = 44100
        pad_end_ms = 500
        pad = int(sr * (pad_end_ms / 1000.0))
        sil = torch.zeros((wav_tensor.shape[0], pad), dtype=wav_tensor.dtype, device=wav_tensor.device)
        wav_tensor = torch.cat([wav_tensor, sil], dim=-1)

        # --- 말미 페이드아웃 ---
        wav_tensor = apply_fade_out(wav_tensor, sr=sr, fade_ms=120.0)

        # 파일 wav로 저장
        now = datetime.datetime.now()
        filename = f"tts_{persona_name}_{now.strftime('%m%d_%H%M%S')}.wav"
        local_path = f"/tmp/{filename}"
        torchaudio.save(local_path, wav_tensor.cpu(), sr, format="wav")

        # S3 업로드
        s3.upload_file(local_path, S3_BUCKET, f"{PREFIX}/{filename}")
        url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": S3_BUCKET, "Key": f"{PREFIX}/{filename}"},
            ExpiresIn=360000  # url 유효기간
        )

        end_time = time.time()

        return {
            "output":{
                "persona": persona_name,
                "text": text,
                "s3_url": url,
                "execution_time": round(end_time - start_time, 2),
                "cwd": os.getcwd()
            }
        }
    except Exception as e:
        import traceback
        return {
            "output":{
                "error": str(e),
                "traceback": traceback.format_exc(),
                "cwd": os.getcwd()
            }
        }


serverless.start({"handler": handler})

# if __name__ == "__main__":
#     # 로컬 테스트용 코드만 여기서 실행 (RunPod에서는 실행 안 됨)
#     import torchaudio
#     wav, sr = torchaudio.load("sample_file/joowoojae.m4a")
#     emb = model.make_speaker_embedding(wav, sr)
#     print("embedding shape:", emb.shape)
