import os, time, datetime, boto3, torch, torchaudio
from runpod import serverless
from zonos.utils import DEFAULT_DEVICE
from zonos.conditioning import make_cond_dict
from zonos.model import Zonos
# from voice_embedding import model

# espeak 변수
os.environ["PHONEMIZER_ESPEAK_PATH"] = "/usr/bin/espeak-ng"

# AWS 환경 변수
S3_BUCKET = os.getenv("AWS_S3_BUCKET")
REGION = os.getenv("AWS_REGION", "ap-northeast-2")
PREFIX = os.getenv("S3_FOLDER_PREFIX", "tts")

# --- 모델 로딩 ---
# HuggingFace에서 로드
# model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=DEFAULT_DEVICE)

# 로컬(도커이미지)에 저장된 모델 불러오기
model = Zonos.from_local(
    config_path="Zonos-v0.1-transformer/config.json",
    model_path="Zonos-v0.1-transformer/model.safetensors",
    device=DEFAULT_DEVICE
)

s3 = boto3.client("s3", region_name=REGION)

# 기본 persona
DEFAULT_SPEAKER = "HongJinkyeong"
DEFAULT_PATH = f"persona_list/{DEFAULT_SPEAKER}"
default_emb = torch.load(DEFAULT_PATH).to(DEFAULT_DEVICE)


def estimate_tokens(text: str) -> int:
    return max(100, len(text) * 20)

def handler(job):
    start_time = time.time()

    text = job["input"].get("text", "안녕하세요")
    persona = job["input"].get("persona", DEFAULT_SPEAKER)

    # persona embedding 불러오기
    speaker_path = f"persona_list/{persona}.pt"
    emb = torch.load(speaker_path).to(DEFAULT_DEVICE) if os.path.exists(speaker_path) else default_emb

    # conditioning
    cond = make_cond_dict(text=text, speaker=emb, language="ko")
    if isinstance(cond["espeak"], tuple):  # espeak 강제 batch=1
        t, l = cond["espeak"]
        cond["espeak"] = ([t[0]], [l[0]])

    # prefix 준비
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        prefix = model.prepare_conditioning(cond)

    max_tokens = estimate_tokens(text)
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

    # 파일 mp3로 저장 (웹 구동엔 mp3가 더 좋음)
    now = datetime.datetime.now()
    filename = f"tts_{persona}_{now.strftime('%m%d_%H%M%S')}.wav"
    local_path = f"/tmp/{filename}"
    torchaudio.save(local_path, wav.cpu(), 44100, format="wav")

    # S3 업로드
    s3.upload_file(local_path, S3_BUCKET, f"{PREFIX}/{filename}", ExtraArgs={"ContentType": "audio/wav"})
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
        "execution_time": round(end_time - start_time, 2)
    }
    # 왜 이걸 안쏘지?
    # 


serverless.start({"handler": handler})

# if __name__ == "__main__":
#     # 로컬 테스트용 코드만 여기서 실행 (RunPod에서는 실행 안 됨)
#     import torchaudio
#     wav, sr = torchaudio.load("sample_file/joowoojae.m4a")
#     emb = model.make_speaker_embedding(wav, sr)
#     print("embedding shape:", emb.shape)
