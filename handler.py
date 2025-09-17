import os, time, datetime, boto3, torch, torchaudio
import librosa, re
from runpod import serverless
from zonos.utils import DEFAULT_DEVICE
from zonos.conditioning import make_cond_dict
from zonos.model import Zonos

# espeak 변수
os.environ["PHONEMIZER_ESPEAK_PATH"] = "/usr/bin/espeak-ng"

# AWS 환경 변수
S3_BUCKET = os.getenv("AWS_S3_BUCKET")
REGION = os.getenv("AWS_REGION", "ap-northeast-2")
PREFIX = os.getenv("S3_FOLDER_PREFIX", "tts")

# --- 모델 로딩 경로 해소 유틸 ---
def _resolve_model_paths():
    candidates = [
        ("Zonos-v0.1-transformer/config.json", "Zonos-v0.1-transformer/model.safetensors"),
        ("/app/Zonos-v0.1-transformer/config.json", "/app/Zonos-v0.1-transformer/model.safetensors"),
        ("/Zonos-v0.1-transformer/config.json", "/Zonos-v0.1-transformer/model.safetensors"),
    ]
    for cfg, mdl in candidates:
        if os.path.exists(cfg) and os.path.exists(mdl):
            return cfg, mdl
    raise FileNotFoundError(
        "Zonos-v0.1-transformer 모델 파일을 찾을 수 없습니다. "
        "컨테이너 내에 다음 경로 중 하나에 있어야 합니다: "
        "./Zonos-v0.1-transformer, /app/Zonos-v0.1-transformer, /Zonos-v0.1-transformer"
    )

print(">>> CWD:", os.getcwd())
try:
    print(">>> /app 목록:", os.listdir("/app"))
except Exception:
    pass

cfg_path, mdl_path = _resolve_model_paths()

# --- 모델 로딩 ---
# HuggingFace에서 로드
# model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=DEFAULT_DEVICE)

# 로컬(도커이미지)에 저장된 모델 불러오기
model = Zonos.from_local(
    config_path=cfg_path,
    model_path=mdl_path,
    device=DEFAULT_DEVICE
)

s3 = boto3.client("s3", region_name=REGION)

# 기본 persona
DEFAULT_SPEAKER = "HongJinkyeong"
DEFAULT_PATH = f"persona_list/{DEFAULT_SPEAKER}.pt"
default_emb = torch.load(DEFAULT_PATH).to(DEFAULT_DEVICE)

# --- 문장 단위 split 함수 ---
def split_sentences(text: str):
    sentences = re.split(r'(?<=[.?!?,])', text)
    return [s.strip() for s in sentences if s.strip()]


def handler(job):
    try:
        start_time = time.time()

        text = job["input"].get("text", "안녕하세요")
        persona = job["input"].get("persona", DEFAULT_SPEAKER)

        # persona embedding 불러오기
        speaker_path = f"persona_list/{persona}.pt"
        emb = torch.load(speaker_path).to(DEFAULT_DEVICE) if os.path.exists(speaker_path) else default_emb

        final_wavs = []

        # 문장 단위로 나눠서 처리
        for sent in split_sentences(text):
            # conditioning
            cond = make_cond_dict(text=sent, speaker=emb, language="ko")
            if isinstance(cond["espeak"], tuple):  # espeak 강제 batch=1
                t, l = cond["espeak"]
                cond["espeak"] = ([t[0]], [l[0]])

            # prefix 준비
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                prefix = model.prepare_conditioning(cond)

                # --- 문장 길이에 따른 max_new_tokens 계산 ---
                max_tokens = max(64, len(sent) * 24)

                # 코드 생성 & 오디오 복원
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

        # --- 무음 제거 ---
        wav_np = wav_full.cpu().numpy()
        wav_trimmed, _ = librosa.effects.trim(wav_np, top_db=30)
        wav_tensor = torch.tensor(wav_trimmed)

        # torchaudio.save 용 shape 보정
        if wav_tensor.ndim == 1:
            wav_tensor = wav_tensor.unsqueeze(0)        # [1, T] (mono)
        elif wav_tensor.ndim == 2:
            pass                                        # [C, T] (stereo 등)
        else:
            raise ValueError(f"Unexpected wav shape after trim: {wav_tensor.shape}")

        # 파일 wav로 저장
        now = datetime.datetime.now()
        filename = f"tts_{persona}_{now.strftime('%m%d_%H%M%S')}.wav"
        local_path = f"/tmp/{filename}"
        torchaudio.save(local_path, wav_tensor.cpu(), 44100, format="wav")

        # S3 업로드
        s3.upload_file(local_path, S3_BUCKET, f"{PREFIX}/{filename}")
        url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": S3_BUCKET, "Key": f"{PREFIX}/{filename}"},
            ExpiresIn=3600  # url 유효기간
        )

        end_time = time.time()

        return {
            "persona": persona,
            "text": text,
            "s3_url": url,
            "execution_time": round(end_time - start_time, 2),
            "cwd": os.getcwd(),
            "model_paths": {"config": cfg_path, "weights": mdl_path}
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "cwd": os.getcwd(),
            "exists": {
                "./cfg": os.path.exists("Zonos-v0.1-transformer/config.json"),
                "/app/cfg": os.path.exists("/app/Zonos-v0.1-transformer/config.json"),
                "/cfg": os.path.exists("/Zonos-v0.1-transformer/config.json"),
            }
        }


serverless.start({"handler": handler})

# if __name__ == "__main__":
#     # 로컬 테스트용 코드만 여기서 실행 (RunPod에서는 실행 안 됨)
#     import torchaudio
#     wav, sr = torchaudio.load("sample_file/joowoojae.m4a")
#     emb = model.make_speaker_embedding(wav, sr)
#     print("embedding shape:", emb.shape)
