from flask import Flask, Response, request
import time

from tool_definitions import tool_definitions
from chatbot import PandaChatBot, LLAMA_32
from groq_bot import GroqChatBot

setup_prompt = "You are a calculator assistant. Use the calculate function to perform mathematical operations and provide the results."

app = Flask(__name__)
bot = GroqChatBot(setup_prompt=setup_prompt, tools=tool_definitions)


# bot = PandaChatBot(LLAMA_32)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/get-response')
def generate_response_stream():
    prompt = request.args.get('prompt')
    is_tool_response = request.args.get('is_tool_response')
    return Response(bot.generate_chat_response(user_input=prompt, tool_response=is_tool_response),
                    mimetype="text/event-stream")
