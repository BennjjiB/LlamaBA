import gradio as gr
from PIL.ImageOps import scale
from gradio import ChatMessage
from dataclasses import dataclass, field
import numpy as np
import threading

import vad
from client import Client
from transcriber import Transcriber

client = Client("http://127.0.0.1:5000", None)
transcriber = Transcriber()


def interact_with_pandabot(prompt, messages):
    messages.append(ChatMessage(role="user", content=prompt))
    yield "", messages
    response = client.send_prompt(prompt)
    messages.append(ChatMessage(role="assistant", content=""))
    for chunk in client.handle_response(response):
        if chunk.get("text"):
            messages[-1] = ChatMessage(role="assistant", content=chunk["text"])
            yield "", messages
        elif chunk.get("tool"):
            messages.pop()
            for tool in chunk["tool"]:
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=f"{tool}",
                        metadata={"title": f"🛠️ Used tool {tool["function_name"]}"}
                    )
                )
            messages.append(ChatMessage(role="assistant", content=""))
        yield "", messages


def capture_audio(new_chunk, transcript, messages):
    if new_chunk:
        new_transcript, stopped_speech = transcriber.transcribe_audio(new_chunk, transcript)
        if stopped_speech and not new_transcript:
            yield from interact_with_pandabot(transcript, messages)
        else:
            yield transcript + new_transcript, messages
    else:
        yield transcript, messages


with gr.Blocks() as demo:
    gr.Markdown("# Chat with a Panda Bot")
    chatbot = gr.Chatbot(
        type="messages",
        label="Agent",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/141/parrot_1f99c.png",
        ),
        layout="bubble"
    )
    text_input = gr.Textbox(
        lines=3,
        label="Chat Message",
        placeholder="Placeholder"
    )
    input_audio = gr.Audio(
        sources=["microphone"],
        label="Input Audio",
        waveform_options=gr.WaveformOptions(waveform_color="#B83A4B"),
        streaming=True,
        type="numpy"
    )
    with gr.Row():
        clear = gr.ClearButton([text_input, chatbot])
        submit_button = gr.Button("Submit", variant="primary", size="lg")

    input_audio.stream(
        fn=capture_audio,
        inputs=[input_audio, text_input, chatbot],
        outputs=[text_input, chatbot],
    )

    submit_button.click(
        fn=interact_with_pandabot,
        inputs=[text_input, chatbot],
        outputs=[text_input, chatbot]
    )
    text_input.submit(interact_with_pandabot, [text_input, chatbot], [text_input, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
