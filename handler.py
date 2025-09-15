from runpod import serverless
import base64
import torchaudio
import torch

# --- Dummy TTS 함수 (테스트용) ---
def dummy_tts(text: str, speaker_id: int):
    # 실제론 zonos TTS 실행할 부분
    # 지금은 1초짜리 사인파 오디오를 생성해봄
    sr = 22050
    t = torch.linspace(0, 1, sr)
    audio = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)  # sine wave (440Hz)

    out_path = "/tmp/out.mp3"
    torchaudio.save(out_path, audio, sr, format="mp3")
    return out_path

# --- RunPod 핸들러 ---
def handler(job):
    text = job["input"].get("text", "안녕하세요")
    speaker_id = int(job["input"].get("speaker_id", 0))

    # TTS 실행
    out_path = dummy_tts(text, speaker_id)

    # 파일을 base64로 인코딩
    with open(out_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "speaker_id": speaker_id,
        "text": text,
        "audio_base64": audio_b64
    }

# RunPod Serverless 등록
serverless.start({"handler": handler})
