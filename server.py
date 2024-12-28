from flask import Flask, Response, request
import time

from chatbot import PandaChatBot, LLAMA_32
from groq_bot import GroqChatBot

app = Flask(__name__)
# bot = GroqChatBot()
bot = PandaChatBot(LLAMA_32)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/get-response')
def generate_response_stream():
    prompt = request.args.get('prompt')
    return Response(bot.generate_chat_response(prompt), mimetype="text/event-stream")
