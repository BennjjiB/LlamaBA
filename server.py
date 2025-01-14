from flask import Flask, Response, request, jsonify
from tool_definitions import sort_bricks_definition
from chatbot import LLAMA_31_8, PandaChatBot, LLAMA_32, LLAMA_31_70, LLAMA_33
from groq_bot import GroqChatBot
from transcriber import Transcriber
import numpy as np

# setup_prompt = """
# You are a calculator assistant.
# You can use the 'calculator' function to perform mathematical calculations.
# Give intermediate status updates if provided by the 'calculator' function,
# like the calculator is booting or the calculator stopped.
# """

# setup_prompt = "You're a helpful assistant."

setup_prompt = f"""
You have access to the following functions:

Use the function 'sort_all_bricks' to: Sort all bricks by either color or size. 
{sort_bricks_definition}


If a you choose to call a function ONLY reply in the following format:
<tool_call>{{"function_name": function name, "arguments": dictionary of argument name and its value}}</tool_call>
Do not use variables.

Here is an example,
<tool_call>{{"function_name": sort_all_bricks, "parameters": {{"by_color": true}}}}</tool_call>

Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line

You are a helpful assistant.
"""

app = Flask(__name__)
# bot = GroqChatBot(model="llama-3.3-70b-specdec", setup_prompt=setup_prompt, tools=tool_definitions)
bot = PandaChatBot(LLAMA_31_8, setup_prompt=setup_prompt)
transcriber = Transcriber()


@app.route("/")
def hello_world():
    return "<p>Welcome to the panda bot api!</p>"


@app.route("/clear")
def clear():
    bot.clear_history(setup_prompt)
    transcriber.reset()
    return "cleared history"


@app.route('/get-response')
def generate_response_stream():
    prompt = request.args.get('prompt')
    is_tool_response = request.args.get('is_tool_response')
    return Response(bot.generate_chat_response(user_input=prompt, is_tool_response=is_tool_response),
                    mimetype="text/event-stream")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json()
    audio_data = np.array(data['audio_data'])
    sample_rate = data['sample_rate']
    transcription, sendPrompt = transcriber.transcribe_audio(audio_data, sample_rate)
    return jsonify({"sendPrompt": sendPrompt, "transcription": transcription})
