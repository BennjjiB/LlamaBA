import gradio as gr
from PIL.ImageOps import scale
from gradio import ChatMessage

from client import Client

client = Client("http://127.0.0.1:5000", None)


def interact_with_pandabot(prompt, messages):
    messages.append(ChatMessage(role="user", content=prompt))
    yield messages
    response = client.send_prompt(prompt)
    messages.append(ChatMessage(role="assistant", content=""))
    for chunk in client.handle_response(response):
        if chunk.get("text"):
            messages[-1] = ChatMessage(role="assistant", content=chunk["text"])
            yield messages
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
        yield messages


with gr.Blocks() as demo:
    gr.Markdown("# Chat with a LangChain Agent 🦜⛓️ and see its thoughts 💭")
    chatbot = gr.Chatbot(
        type="messages",
        label="Agent",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/141/parrot_1f99c.png",
        ),
        layout="bubble"
    )

    with gr.Column():
        input = gr.Textbox(lines=1, label="Chat Message", placeholder=
        "Placeholder")
    with gr.Row():
        clear = gr.ClearButton([input, chatbot])
        submit_button = gr.Button("Submit", variant="primary", size="lg")
    submit_button.click(
        fn=interact_with_pandabot,
        inputs=[input, chatbot],
        outputs=[chatbot]
    )
    input.submit(interact_with_pandabot, [input, chatbot], [chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
