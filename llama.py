import os
import json
from typing import Literal
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLAMA_31 = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_32 = "meta-llama/Llama-3.2-1B-Instruct"

print('Running on:', DEVICE)


def getToolDefinitions(tool_file_path: str):
    with open(tool_file_path, "r") as file:
        data = json.load(file)
        return json.dumps(data)


class PandaChatBot:
    def __init__(self, model_path, quantization: Literal["16bit", "8bit", "4bit"] = "16bit"):
        # Quantitation
        if quantization == "16bit":
            self.quantization_config = None
        elif quantization == "8bit":
            self.quantization_config = {"load_in_8bit": True}
        elif quantization == "4bit":
            self.quantization_config = {"load_in_4bit": True}

        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=self.quantization_config
        )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Conversation for history
        self.conversation = []

    def get_response_streamer(
        self, query, max_tokens=1028, temperature=0.6, top_p=0.9
    ):
        self.conversation.append({"role": "user", "content": query})
        # Create tokenized prompt
        prompt = self.tokenizer.apply_chat_template(
            self.conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        # Create text streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            timeout=10,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            inputs=prompt,
            streamer=streamer,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def generate_chat_response(self, user_input: str):
        streamer = self.get_response_streamer(user_input)
        generated_response = ""
        for response in streamer:
            generated_response += response
            yield response
        self.conversation.append(
            {"role": "assistant", "content": generated_response})

    def setup(self, system_instructions):
        self.conversation.append(
            {"role": "system", "content": system_instructions})

    def chatbot(self, system_instructions=""):
        """
        Generates a chatbot interface. Useful for texting the llm.
        """
        self.setup(system_instructions)
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the chatbot. Goodbye!")
                break
            generated_response = ""
            for response in self.generate_chat_response(user_input):
                generated_response += response
                if (response.strip()):
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("User input:", user_input)
                    print("Llama response:", generated_response)
                    print('', end='', flush=True)
            print()

setup_prompt = f"""
            You are a robot arm named Panda with tool calling capabilities.
            Your task is to grab and sort colored blocks. Respond in a positive manner.
            When you receive a tool call response, use the output to format an answer to the original user question.
            If you decide to invoke any of the function(s), you MUST put it in the format of
            [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
            You SHOULD NOT include any other text in the response.
            Here is a list of functions in JSON format that you can invoke.
            {getToolDefinitions("tool_definitions.json")}
        """


if __name__ == "__main__":
    bot = PandaChatBot(LLAMA_32)
    bot.chatbot()
