from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llama import PandaChatBot, LLAMA_32, setup_prompt_1, LLAMA_31_8, LLAMA_31_70

app = FastAPI()
bot = PandaChatBot(LLAMA_31_8)
#bot.setup(setup_prompt_1)


@app.get("/")
def root():
    return "Hello to the llama bot!"

import time

def fake_data_streamer():
    for i in range(10):
        yield b'some fake data\n\n'
        time.sleep(0.5)

@app.get("/get-response")
def get_response(prompt: str):
    # response_generator = bot.generate_chat_response(prompt)
    response_generator = fake_data_streamer()
    return StreamingResponse(response_generator, media_type="text/event-stream")


@app.get("/clear-history")
def clear_history():
    bot.clearHistory()
    return "History cleared"
