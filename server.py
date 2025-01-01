from flask import Flask, Response, request
import time

from tool_definitions import tool_definitions
from chatbot import PandaChatBot, LLAMA_32
from groq_bot import GroqChatBot

setup_prompt = """
You are a calculator assistant.
You can use the 'calculator' function to perform mathematical calculations.
Give intermediate status updates if provided by the 'calculator' function,
like the calculator is booting or the calculator stopped.
"""

# setup_prompt = "You're a helpful assistant."

app = Flask(__name__)
bot = GroqChatBot(setup_prompt=setup_prompt, tools=tool_definitions)


# bot = GroqChatBot(model="llama3-70b-8192", setup_prompt=setup_prompt, tools=None)


# bot = PandaChatBot(LLAMA_32)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/clear")
def clear():
    bot.clear_history()
    return "cleared history"


@app.route('/get-response')
def generate_response_stream():
    prompt = request.args.get('prompt')
    is_tool_response = request.args.get('is_tool_response')
    return Response(bot.generate_chat_response(user_input=prompt, is_tool_response=is_tool_response),
                    mimetype="text/event-stream")
