from flask import Flask, Response, request, jsonify
import tool_definitions
from chatbot import LLAMA_31_8, PandaChatBot, LLAMA_32, LLAMA_31_70, LLAMA_33
from groq_bot import GroqChatBot
from transcriber import Transcriber
import numpy as np
import inquirer


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
{tool_definitions.sort_bricks_definition}


Use the function 'grab_brick' to: Grab and sort one brick specified by its color. 
{tool_definitions.grab_brick}


Use the function 'get_collision_free_bricks' to: Get a list of the size and color of all collision free bricks.
{tool_definitions.get_collision_free_bricks}


Use the function 'get_all_bricks' to: Get a list of all bricks, visible to the robot. Those bricks might not be collision free.
{tool_definitions.get_all_bricks}

Here is an example,
user: How many bricks can you see?
assistant: <tool_call>{{"function_name": "get_all_bricks", "arguments": {{}}}}</tool_call>
ipython: [('4x2', 'orange'), ('4x2', 'green'), ('4x2', 'blue')]
assistant: I can see 1 orange brick with size 4x2, one green brick with size 4x2 and one blue brick with size 4x2.


If a you choose to call a function ONLY reply in the following format:
<tool_call>{{"function_name": function name, "arguments": dictionary of argument name and its value}}</tool_call>
Do not use variables.

Here is an example,
<tool_call>{{"function_name": "sort_all_bricks", "arguments": {{"by_color": true}}}}</tool_call>

Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Put the entire function call reply on one line

You are a helpful assistant. Your name is Panda. 
"""

app = Flask(__name__)
# bot = GroqChatBot(model="llama-3.3-70b-specdec", setup_prompt=setup_prompt, tools=tool_definitions)


def get_llama_version() -> str:
    questions = [
        inquirer.List('Llama Model',
                      message="What llama model should be used.",
                      choices=['3.1 8B', '3.3 70B'],
                      carousel=True
                      )
    ]
    result = inquirer.prompt(questions)
    if result == '3.1 8B':
        return LLAMA_31_8
    elif result == '3.3 70B':
        return LLAMA_33


bot = PandaChatBot(get_llama_version(), setup_prompt=setup_prompt)
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
    transcription, sendPrompt = transcriber.transcribe_audio(
        audio_data, sample_rate)
    return jsonify({"sendPrompt": sendPrompt, "transcription": transcription})
