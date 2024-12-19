import pyaudio
import wave

# Parameters for recording
FORMAT = pyaudio.paInt16   # Format of the audio (16-bit PCM)
CHANNELS = 1               # Mono audio
RATE = 44100               # Sample rate (samples per second)
CHUNK = 1024               # Size of each audio chunk (how much data to read at a time)
RECORD_SECONDS = 5         # Duration of the recording
OUTPUT_FILENAME = "recorded_audio.wav"  # Output filename

# Initialize the PyAudio object
p = pyaudio.PyAudio()

# Open a stream for recording
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Recording...")

# Record the audio
frames = []
for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

# Stop the stream and close it
stream.stop_stream()
stream.close()

# Terminate PyAudio
p.terminate()

# Save the recorded audio to a WAV file
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Audio saved to {OUTPUT_FILENAME}")
