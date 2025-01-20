import numpy as np
from faster_whisper import WhisperModel
from scipy.signal import resample

import time


class Transcriber():
    def __init__(
            self,
            model_type="large-v3",
            device="cuda",
            compute_type="float16",
            start_prompt_delay=2
    ):
        """
        model_type: whisper model type (tiny, base, small, medium, large, turbo)
        device: use "cpu" or "cuda"
        compute_type: use "int8" or "float16"
        """
        self.whisper = WhisperModel(
            model_type, device=device, compute_type=compute_type)
        self.buffer = np.array([], dtype=np.float32)
        self.started_speaking = False
        self.stopped_speaking_time = None
        self.start_prompt_delay = start_prompt_delay
        self.sentences = []
        self.old_transcript = ""

    def transcribe_audio(self, audio_data, sr):
        if self.stopped_speaking_time is not None:
            current_time = time.time()
            time_diff = current_time - self.stopped_speaking_time
            if time_diff >= self.start_prompt_delay:
                print("Should sent prompt")
                self.sentences = []
                prompt = self.old_transcript
                self.old_transcript = ""
                self.stopped_speaking_time = None
                self.reset()
                return prompt, True

        cleaned_audio = self.__clean_audio(sr, audio_data)
        self.__update_buffer(cleaned_audio)
        new_transcript = self.__transcribe(self.buffer)
        if (not self.sentences or new_transcript != self.sentences[-1]) and new_transcript:
            if not self.started_speaking:
                self.started_speaking = True
                self.stopped_speaking_time = None
                print("Started speaking", new_transcript)
                self.sentences.append(new_transcript)
            else:
                self.sentences[-1] = new_transcript
            self.old_transcript = "\n".join(self.sentences)
            return "\n".join(self.sentences), False
        else:
            self.reset()
            if self.started_speaking:
                self.stopped_speaking_time = time.time()
                self.started_speaking = False
                print("Stopped speaking")
            return self.old_transcript, False

    def reset(self):
        self.buffer = np.array([], dtype=np.float32)

    def hard_reset(self):
        self.old_transcript = ""
        self.sentences = []
        self.buffer = np.array([], dtype=np.float32)

    def transcribe(self, audio_data) -> str:
        # add vad_filter
        segments, _ = self.whisper.transcribe(
            audio_data,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                onset=0.9, offset=0.9 - 0.15, min_speech_duration_ms=0, min_silence_duration_ms=1000)
        )
        segments = list(segments)
        return "".join([segment.text for segment in segments]).strip()

    def clean_audio(self, sr, audio_data):
        # Convert to mono if stereo
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        num_samples = round(len(audio_data) * float(16000) / sr)
        # Resample the audio data to the target sample rate
        resampled_audio = resample(audio_data, num_samples)
        audio_data = resampled_audio.astype(np.float32)
        # Normalize the audio (if necessary)
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        return audio_data

    def __update_buffer(self, chunk):
        if len(self.buffer) / 16000 > 20:
            self.buffer = np.array([], dtype=np.float32)
        self.buffer = np.concatenate((self.buffer, chunk))