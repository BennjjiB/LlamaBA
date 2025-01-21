import soundfile as sf
from transcriber import Transcriber
from jiwer import wer

SAMPLE_SIZE = 1

transcriber = Transcriber()
transcriptions = []
targets = []

# Observation
for i in range(0, SAMPLE_SIZE):
    file_path = f"test_audio/{i}.wav"
    target_path = f"test_audio/{i}.txt"
    audio_data, sample_rate = sf.read(file_path)
    audio_data = transcriber.clean_audio(sample_rate, audio_data)
    transcription = transcriber.transcribe(audio_data)
    transcriptions.append(transcription)
    with open(target_path, 'r') as file:
        target = file.readlines()
        targets.append(target)
    print(transcription)

# Evaluation
word_error_rate = wer(targets, transcriptions)

# Metrics: 
# Speed Factor = Audio Duration (Real Time) / Transcription Time
# Word Error Rate = Edit distance (https://en.wikipedia.org/wiki/Word_error_rate)