import soundfile as sf
from transcriber import Transcriber

SAMPLE_SIZE = 1

transcriber = Transcriber()

for i in range(0, SAMPLE_SIZE):
    file_path = f"test_audio/{i}.wav"
    audio_data, sample_rate = sf.read(file_path)
    audio_data = transcriber.clean_audio(sample_rate, audio_data)
    transcription = transcriber.transcribe(audio_data)
    print(transcription)



