import io
import pyaudio
from pydub import AudioSegment
import numpy as np 

class AudioProcessor:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels

    def convert_audio(self, audio_segment):
        """Convert AudioSegment to the desired parameters (sample rate, channels)."""
        return audio_segment.set_frame_rate(self.sample_rate).set_channels(self.channels).set_sample_width(2)
    
    def combine_audio(self, audio_segments):
        """Combine all loaded audio segments into one."""
        combined_audio = AudioSegment.silent(duration=0)  # Initialize with silent audio
        for audio in audio_segments:
            audio = self.convert_audio(audio)
            combined_audio += audio  # Concatenate audio segments
        return combined_audio


    def process(self, wav_data_list):
        """Load, combine, and play audio from a list of WAV data."""
        # Load the audio data
        segments = []
        for wav_data in wav_data_list:
            segments.append(AudioSegment.from_wav(io.BytesIO(wav_data)))

        # Combine the audio
        combined_audio = self.combine_audio(segments)

        # Convert combined audio back to raw audio bytes
        # Convert buffer to float32 using NumPy                                                                                 
        audio_as_np_int16 = np.frombuffer(combined_audio.raw_data, dtype=np.int16)
        audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

        # Normalise float32 array so that values are between -1.0 and +1.0                                                      
        max_int16 = 2**15
        audio_normalised = audio_as_np_float32 / max_int16
        return audio_normalised


            
