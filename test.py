import speech_recognition as sr
import time

# Define a callback function to handle the audio recording
def callback(_, audio):
    print("Audio recorded successfully.")
    try:
        # Save the audio to a WAV file
        with open("recorded_audio.wav", "wb") as file:
            file.write(audio.get_wav_data())
        print("Audio saved to 'recorded_audio.wav'.")
    except Exception as e:
        print(f"Error saving audio: {e}")

recognizer = sr.Recognizer()
print("Available microphones:", sr.Microphone.list_microphone_names())

# Define the microphone device index
try:
    # Start listening in the background using the device_index
    print("Say something:")
    # Start listening in the background
    stop_listening = recognizer.listen_in_background(sr.Microphone(device_index=10), callback)
    
    # Main thread can do other tasks, here we let it run for 10 seconds for demo purposes
    time.sleep(10)  # Wait for a while to let background listening happen
    
    # Stop the background listening after 10 seconds
    stop_listening(wait_for_stop=True)
    print("Background listening stopped.")
        
except Exception as e:
    print(f"Error: {e}")