# Transcription and LLM Server
This server contains a faster whisper model for live transcription and a Llama model for chatting and tool calls.

## Transcription
For transcription [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) is used. 
1. Audio is cleaned and converted to a mono channel with sample rate 16kHz.
2. Audio is added to a buffer array
3. Faster Whisper transcribes the audio: 
    - The build in VAD (SileroVAD) is used to detect speech segments and will only give those to the whisper model for transcription
4. The transcription is streamed back to the client. 
5. If the transcription does not change for 2s (start_prompt_delay) of audio, it is assumed the speaker stopped. At this point the whole transcription can be sent to the llm and the buffer will be cleared. 

## LLM
For the llm [Llama](https://huggingface.co/meta-llama) is used.
Especially the 3.1 8B and 3.3 70B Models. 
Huggingface transformers with chat templates are used to tokenize prompts and stream the generated response. 

There is also a version using the [Groq](https://groq.com/) api. This is good for testing proposes 
