import time
import soundfile as sf
from experiment_prompts import easy_prompt, neutral_prompt, hard_prompt
from transcriber import Transcriber
from jiwer import wer
import inquirer

# Define your Whisper models as constants (example names, replace with actual model variables)
WHISPER_TINY = "tiny"
WHISPER_BASE = "base"
WHISPER_SMALL = "small"
WHISPER_MEDIUM = "medium"
WHISPER_LARGE = "large-v3"


def get_whisper_model():
    # First question: Select the Whisper model
    model_question = [
        inquirer.List('Whisper Model',
                      message="Which Whisper model should be used?",
                      choices=['tiny', 'base', 'small', 'medium', 'large'],
                      carousel=True
                      )
    ]
    model_result = inquirer.prompt(model_question).get("Whisper Model", 'tiny')

    # Map model_result to the corresponding constant
    if model_result == 'tiny':
        model_constant = WHISPER_TINY
    elif model_result == 'base':
        model_constant = WHISPER_BASE
    elif model_result == 'small':
        model_constant = WHISPER_SMALL
    elif model_result == 'medium':
        model_constant = WHISPER_MEDIUM
    elif model_result == 'large':
        model_constant = WHISPER_LARGE
    else:
        model_constant = WHISPER_TINY  # Fallback to 'tiny' as default
    return model_constant, model_result


SAMPLE_SIZE = 54
noise_samples = [16, 35, 23, 3, 8, 7, 5,
                 33, 2, 1, 25, 28, 32, 20, 17, 21, 14, 9]

model_constant, model_result = get_whisper_model()

transcriber = Transcriber(model_type=model_result)
transcriptions = []
targets = easy_prompt + neutral_prompt + hard_prompt + [""]*18
duration = 0
transcription_time = 0

for language in ["en", None]:
    transcriber.language = language
    # Observation
    for i in range(0, SAMPLE_SIZE-1):
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
    word_error_rate_clean = wer([targets[i] for i in range(36) if i not in noise_samples],
                                [transcriptions[i] for i in range(36) if i not in noise_samples])
    word_error_rate_noise = wer([targets[i] for i in noise_samples], [
                                transcriptions[i] for i in noise_samples])
    word_error_rate_only_noise = wer([targets[i] for i in range(36, 55)],
                                     [transcriptions[i] for i in range(36, 55)])
    word_error_total = wer(targets, transcriptions)
    # Speed Factor = Audio Duration (Real Time) / Transcription Time
    speed = duration / transcription_time

    results = "Transcription Results for {model_constant} and language {language}:\n-----------------------\n"
    results += f"""
        Metrics:
        --------
        Accuracy Total: {1 - word_error_total:.4f}
        Accuracy Clean: {1 - word_error_rate_clean:.4f}
        Accuracy text with Noise: {1 - word_error_rate_noise:.4f}
        Accuracy only with Noise: {1 - word_error_rate_noise:.4f}
        Speed (seconds of audio per second of processing): {speed:.4f}
        Total Audio Duration: {duration:.2f} seconds
        Total Transcription Time: {transcription_time:.2f} seconds 
        \n
    """
    for i, (target, transcription) in enumerate(zip(targets, transcriptions)):
        results += f"Sample {i + 1}:\nTarget: {target}\nTranscription: {transcription}\n Noise: {i in noise_samples}\n\n"

# Save results to a file
output_file = f"transcription_results_{model_constant}.txt"
with open(output_file, "w") as file:
    file.write(results)

print(word_error_rate_clean, speed)
