import os
import asyncio
from datetime import datetime, timezone, timedelta
from queue import Queue
from sys import platform

import numpy as np
import speech_recognition as sr
from faster_whisper import WhisperModel


class Transcriber():
    """
    A class to handle real-time audio transcription using the Faster Whisper model.
    """

    def __init__(
        self,
        model_type="tiny.en",
        device="cpu",
        compute_type="int8",
        energy_threshold=1000,
        default_microphone='list'
    ):
        """
        model_type: whisper model type (tiny, base, small, medium, large, turbo)
        device: use "cpu" or "cuda"
        compute_type: use "int8" or "float16"
        energy_threshold: Energy level for mic to detect speech
        default_microphone: Microphone name for SpeechRecognition. Use 'list' to see all available options
        """
        self.__setup_mic(default_microphone)
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

    async def start_transcription(
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
            data = audio.get_raw_data()
            self.data_queue.put(data)

        # Create a background thread that will pass us raw audio bytes.
        self.recorder.listen_in_background(
            self.source, record_callback, phrase_time_limit=record_timeout)
            
        while True:
            try:
                now = datetime.now(timezone.utc)
                if phrase_time and now - phrase_time > timedelta(seconds=speak_timeout):
                    yield transcription
                    transcription = ['']
                    phrase_time = None
                # Pull raw recorded audio from the queue.
                if not self.data_queue.empty():
                    phrase_complete = False
                    # If enough time has passed between recordings, consider the phrase complete.
                    # Clear the current working audio buffer to start over with the new data.
                    if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                        phrase_complete = True
                    phrase_time = now

                    # Combine audio data from queue
                    audio_data = b''.join(self.data_queue.queue)
                    self.data_queue.queue.clear()

                    text = self.__transcribe(audio_data)

                    if phrase_complete:
                        transcription.append(text)
                    else:
                        transcription[-1] += text

                    # Clear the console to reprint the updated transcription.
                    os.system('cls' if os.name == 'nt' else 'clear')
                    for line in transcription:
                        print(line)
                    # Flush stdout.
                    print('', end='', flush=True)
                else:
                    # Infinite loops are bad for processors, must sleep.
                    await asyncio.sleep(0.25)
            except asyncio.CancelledError:
                break
            except KeyboardInterrupt:
                break

    def __transcribe(self, audio_data: bytes) -> str:
        """
        Transcribes some audio data and returns the result as a string 
        audio_data: The raw audio data in bytes 
        """
        # Convert in-ram buffer to something the model can use directly without needing a temp file.
        # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
        # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
        audio_np = np.frombuffer(
            audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        segments, _ = self.whisper.transcribe(audio_np, beam_size=5)
        segments = list(segments)
        text = "".join(segment.text for segment in segments)
        return text

    def __setup_mic(self, default_microphone) -> sr.Microphone:
        """
        Finds and returns the speech recognition microphone 
        default_microphone: Microphone name for SpeechRecognition. Use 'list' to see all available options
        """
        # Prevents permanent application hang and crash by using the wrong Microphone
        if 'linux' in platform:
            mic_name = default_microphone
            if not mic_name or mic_name == 'list':
                print("Available microphone devices are: ")
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    print(f"Microphone with name \"{name}\" found")
                return
            else:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        self.source = sr.Microphone(
                            sample_rate=16000, device_index=index)
        else:
            self.source = sr.Microphone(sample_rate=16000)


async def main():
    transcriber = Transcriber()
    async for transcription in transcriber.start_transcription():
        print("\n Yielded line:", "".join(transcription))

if __name__ == "__main__":
    asyncio.run(main())