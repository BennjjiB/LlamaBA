from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from time import sleep

app = FastAPI()


@app.get("/")
def root():
    return "Hello to the llama bot!"


def generateResponse(prompt: str):
    story_parts = [
        "Once upon a time, in a land far away,\n",
        "a young boy named Liam found a hidden map.\n",
        "Following the map, he discovered a magical garden.\n",
        "There, he found a key that opened a secret door.\n",
        "From then on, Liam protected the magical garden.\n"
    ]
    for part in story_parts:
        yield part
        sleep(0.5)


@app.get("/get-response")
def get_response(prompt: str):
    return StreamingResponse(generateResponse(prompt), media_type="text/event-stream")
