from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
from asyncio import sleep

app = FastAPI()


@app.get("/")
def root():
    return "Hello to the llama bot!"


async def generateResponse(prompt: str) -> AsyncGenerator:
    response_iter = [prompt, "wjkhadhassdh", "adasda", "kljyjfdslakjdl"]
    for response in response_iter:
        yield f"data: {response}\n\n"
        await sleep(1)


@app.get("/get-response")
def get_response(prompt: str):
    return StreamingResponse(generateResponse(prompt), media_type="text/event-stream")
