from flask import Flask, Response, request, jsonify
from setup import get_llama_version, setup_prompt, get_language, get_whisper_model
from chatbot import PandaChatBot
from transcriber import Transcriber
import numpy as np

app = Flask(__name__)
lama_v = get_llama_version()
language = get_language()
whisper_v = get_whisper_model()
bot = PandaChatBot(lama_v, setup_prompt=setup_prompt)
transcriber = Transcriber(model_type=whisper_v, language=language)


@app.route("/")
def hello_world():
    return "<p>Welcome to the panda bot api!</p>"


@app.route("/clear")
def clear():
    bot.clear_history(setup_prompt)
    transcriber.hard_reset()
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
