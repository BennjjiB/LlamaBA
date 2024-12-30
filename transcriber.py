import os
from audio_processor import AudioProcessor
from datetime import datetime, timezone, timedelta
from queue import Queue
from sys import platform
from time import sleep
import numpy as np
import speech_recognition as sr
from faster_whisper import WhisperModel
import soundfile as sf
from scipy.signal import resample
import noisereduce as nr


class Transcriber():
    def __init__(
            self,
            model_type="tiny.en",
            device="cpu",
            compute_type="int8",
            max_audio_chunck_duration=2
    ):
        """
        model_type: whisper model type (tiny, base, small, medium, large, turbo)
        device: use "cpu" or "cuda"
        compute_type: use "int8" or "float16"
        """
        self.whisper = WhisperModel(
            model_type, device=device, compute_type=compute_type)
        self.max_samples = int(max_audio_chunck_duration * 16000)
        self.buffer = np.array([], dtype=np.float32)
        self.started_speaking = False

    def __transcribe(self, audio_data) -> str:
        # add vad_filter
        segments, _ = self.whisper.transcribe(
            audio_data,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(threshold=0.9, min_speech_duration_ms=500, min_silence_duration_ms=2000)
        )
        segments = list(segments)
        text = "".join(segment.text for segment in segments).strip()
        return text

    def transcribe_audio(self, chunk, transcription):
        sr, audio_data = chunk
        cleaned_audio = self.__clean_audio(sr, audio_data)
        self.__update_buffer(cleaned_audio)
        response = ""
        response = self.__transcribe(self.buffer)
        stopped = False
        if self.started_speaking and not response:
            self.reset()
            stopped = True
            print("Stopped speaking")
        elif not self.started_speaking and response:
            self.started_speaking = True
            print("Started speaking", response)
        return response, stopped

    def reset(self):
        self.started_speaking = False
        self.buffer = np.array([], dtype=np.float32)

    def __clean_audio(self, sr, audio_data):
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
        reduced_noise_samples = nr.reduce_noise(y=audio_data, sr=sr)
        return reduced_noise_samples

    def __update_buffer(self, new_chunk):
        self.buffer = np.concatenate((self.buffer, new_chunk))
        if len(self.buffer) > self.max_samples:
            self.buffer = self.buffer[-self.max_samples:]


class TranscriberOld():
    """
    A class to handle real-time audio transcription using the Faster Whisper model.
    """

    def __init__(
            self,
            model_type="medium.en",
            device="cpu",
            compute_type="int8",
            energy_threshold=1000,
            default_microphone=''
    ):
        """
        model_type: whisper model type (tiny, base, small, medium, large, turbo)
        device: use "cpu" or "cuda"
        compute_type: use "int8" or "float16"
        energy_threshold: Energy level for mic to detect speech
        default_microphone: Microphone name for SpeechRecognition. Use 'list' to see all available options
        """
        print("Setting up whisper...")
        self.source = sr.Microphone()
        self.audio_processor = AudioProcessor()
        self.whisper = WhisperModel(
            model_type, device=device, compute_type=compute_type)
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = energy_threshold
        # Dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
        self.recorder.dynamic_energy_threshold = False
        # Thread safe Queue for passing data from the threaded recording callback.
        self.data_queue = Queue()
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

    def start_transcription(
            self,
            record_timeout=3,
            phrase_timeout=3,
            speak_timeout=5
    ):
        """
        Generator function, creating audio transcriptions.
        record_timeout: The maximum length in seconds for one recording
        phrase_timeout: Time between recordings before it is considered as a new sentence 
        speak_timeout: The maximum length of silence after which the current transcription will be yielded 
        """
        # The last time a recording was retrieved from the queue.
        phrase_time = None
        transcription = ['']

        def record_callback(_, audio: sr.AudioData) -> None:
            """
            Threaded callback function to receive audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
            """
            # Grab the raw bytes and push it into the thread safe queue.
            data = audio.get_wav_data()
            self.data_queue.put(data)

        # Create a background thread that will pass us raw audio bytes.
        stop_listening = self.recorder.listen_in_background(
            self.source, record_callback, phrase_time_limit=record_timeout)

        print("Speak to Panda Bot:")
        while True:
            try:
                now = datetime.now(timezone.utc)
                # if phrase_time and now - phrase_time > timedelta(seconds=speak_timeout) and self.data_queue.empty():
                #     stop_listening()
                #     return transcription
                # Pull raw recorded audio from the queue.
                if not self.data_queue.empty():
                    phrase_complete = False
                    # If enough time has passed between recordings, consider the phrase complete.
                    # Clear the current working audio buffer to start over with the new data.
                    if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                        phrase_complete = True
                    phrase_time = now
                    # Combine audio data from queue
                    audio_data = self.audio_processor.process(self.data_queue.queue)
                    self.data_queue.queue.clear()

                    text = self.__transcribe(audio_data).strip()

                    if phrase_complete:
                        transcription.append(text)
                    else:
                        transcription[-1] += text

                    # Clear the console to reprint the updated transcription.
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("Speak to Panda Bot:")
                    for line in transcription:
                        print(line)
                    # Flush stdout.
                    print('', end='', flush=True)
                else:
                    # Infinite loops are bad for processors, must sleep.
                    sleep(0.25)
            except KeyboardInterrupt:
                break

    def __transcribe(self, audio_data: bytes) -> str:
        """
        Transcribes some audio data and returns the result as a string 
        audio_data: The raw audio data in bytes 
        """
        segments, _ = self.whisper.transcribe(audio_data, beam_size=5)
        segments = list(segments)
        text = "".join(segment.text for segment in segments)
        return text

    def __setup_mic(self, default_microphone) -> sr.Microphone:
        """
        Finds and returns the speech recognition microphone 
        default_microphone: Microphone name for SpeechRecognition. Use 'list' to see all available options
        """
        mic_name = default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    self.source = sr.Microphone()


def main():
    transcriber = TranscriberOld()
    for transcription in transcriber.start_transcription():
        print("\n Yielded line:", "".join(transcription))
