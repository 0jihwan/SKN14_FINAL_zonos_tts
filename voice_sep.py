# 음성 파일에서 특정 화자의 목소리만 추출하는 코드
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
from dotenv import load_dotenv
import torch

# .env 파일에서 환경 변수 로드
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# GPU(CUDA) 사용 가능 여부 확인
if torch.cuda.is_available():
    device = "cuda"
    print("✅ GPU(CUDA)를 사용하여 작업합니다.")
else:
    device = "cpu"
    print("⚠️ GPU(CUDA)를 찾을 수 없습니다. CPU를 사용하여 작업합니다.")

# Hugging Face 모델 파이프라인 로드
if not HF_TOKEN:
    print("오류: 환경 변수 'HF_TOKEN'을 찾을 수 없습니다.")
    exit()

try:
    # 파이프라인을 특정 장치(GPU 또는 CPU)로 로드
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                       use_auth_token=HF_TOKEN).to(torch.device(device))
except Exception as e:
    print(f"파이프라인 로드 중 오류 발생: {e}")
    print("Hugging Face에서 'pyannote/speaker-diarization' 및 'pyannote/segmentation' 모델의 약관에 동의했는지 확인하세요.")
    exit()

name = "hanhyaejin"
# 원본 오디오 파일 경로
input_file = f"sample_file/{name}1.mp3"

# MP3 파일을 WAV로 변환 (pyannote는 WAV를 권장)
print("🔊 MP3 파일을 WAV로 변환 중...")
audio = AudioSegment.from_mp3(input_file)
wav_file = "temp_converted.wav"
audio.export(wav_file, format="wav")

# 화자 분할 실행
print("🎙️ 화자 분할(Diarization) 시작...")
diarization = pipeline(wav_file)

# 원하는 화자 ID를 설정하세요 (예: SPEAKER_00, SPEAKER_01 등)
target_speaker_id = "SPEAKER_00"

# 특정 화자의 음성만 담을 빈 AudioSegment 생성
target_speaker_audio = AudioSegment.empty()

# 분할된 오디오 구간들을 합치기
print(f"✂️ 화자({target_speaker_id})의 음성 구간 추출 및 병합 중...")
full_audio = AudioSegment.from_file(wav_file)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    if speaker == target_speaker_id:
        start_ms = turn.start * 1000
        end_ms = turn.end * 1000
        segment = full_audio[start_ms:end_ms]
        target_speaker_audio += segment

# 추출된 음성 파일을 WAV로 저장
output_file = f"extracted_audio/{target_speaker_id}_{name}1.wav"
os.makedirs("extracted_audio", exist_ok=True)
target_speaker_audio.export(output_file, format="wav")

# 임시 파일 삭제
os.remove(wav_file)

print(f"✅ 특정 화자({target_speaker_id})의 음성 파일이 성공적으로 저장되었습니다: {output_file}")