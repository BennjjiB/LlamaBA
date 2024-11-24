from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llama import PandaChatBot, LLAMA_32, setup_prompt_1

app = FastAPI()
bot = PandaChatBot(LLAMA_32)
bot.setup(setup_prompt_1)


@app.get("/")
def root():
    return "Hello to the llama bot!"


@app.get("/get-response")
def get_response(prompt: str):
    response_generator = bot.generate_chat_response(prompt)
    return StreamingResponse(response_generator, media_type="text/event-stream")


@app.get("/clear-history")
def clear_history():
    bot.clearHistory()
    return "History cleared"
