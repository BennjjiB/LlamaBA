import time
import soundfile as sf
from scipy.special import target
from trimesh.permutate import noise

from experiment_prompts import easy_prompt, neutral_prompt, hard_prompt
from transcriber import Transcriber
from jiwer import wer

SAMPLE_SIZE = 36
noise_samples = [16, 36, 23, 3, 8, 7, 5, 33, 2, 1, 25, 28, 32, 20, 17, 21]

transcriber = Transcriber()
transcriptions = []
targets = easy_prompt + neutral_prompt + hard_prompt
duration = 0
transcription_time = 0

# Observation
for i in range(0, SAMPLE_SIZE):
    file_path = f"test_audio/{i}.wav"
    target = targets[i]
    f = sf.SoundFile(file_path)
    audio_data, sample_rate = sf.read(file_path)
    duration += f.frames / f.samplerate
    audio_data = transcriber.clean_audio(sample_rate, audio_data)
    start = time.time()
    transcription = transcriber.transcribe(audio_data)
    transcription_time += time.time() - start
    transcriptions.append(transcription)

# Evaluation
# Word Error Rate = Edit distance (https://en.wikipedia.org/wiki/Word_error_rate)
word_error_rate_clean = wer([targets[i] for i in range(len(targets)) if i not in noise_samples],
                            [transcriptions[i] for i in range(len(transcriptions)) if i not in noise_samples])
word_error_rate_noise = wer([targets[i] for i in noise_samples], [transcriptions[i] for i in noise_samples])
word_error_total = wer(targets, transcriptions)
# Speed Factor = Audio Duration (Real Time) / Transcription Time
speed = duration / transcription_time

results = "Transcription Results:\n-----------------------\n"
for i, (target, transcription) in enumerate(zip(targets, transcriptions)):
    results += f"Sample {i + 1}:\nTarget: {target}\nTranscription: {transcription}\n Noise: {i in noise_samples}\n\n"

results += f"""
Metrics:
--------
Word Error Rate (WER) Total: {word_error_rate:.4f}
Word Error Rate (WER) Clean: {word_error_rate:.4f}
Word Error Rate (WER) Noise: {word_error_rate:.4f}
Speed (seconds of audio per second of processing): {speed:.4f}
Total Audio Duration: {duration:.2f} seconds
Total Transcription Time: {transcription_time:.2f} seconds
"""

# Save results to a file
output_file = "transcription_results.txt"
with open(output_file, "w") as file:
    file.write(results)

print(word_error_rate, speed)
