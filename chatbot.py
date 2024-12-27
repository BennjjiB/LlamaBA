import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from tool_definitions import tools
from abc import ABC, abstractmethod
from typing import Literal, List, Dict
import torch

LLAMA_31_8 = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_31_70 = "meta-llama/Llama-3.1-70B-Instruct"
LLAMA_32 = "meta-llama/Llama-3.2-1B-Instruct"


class AbstractChatBot(ABC):
    def __init__(self, setup_prompt: str = "", tools: List = []):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Running on:', self.device)
        self.tools = tools
        self.setup(setup_prompt)

    def setup(self, system_instructions):
        self.conversation = []
        self.conversation.append(
            {"role": "system", "content": system_instructions})

    @abstractmethod
    def get_response_streamer(self, query: str, max_tokens: int = 1028, temperature: float = 0.6,
                              top_p: float = 0.9):
        """
        Returns a response streamer for generating chatbot responses.
        """
        pass

    def generate_chat_response(self, user_input: str, with_memory: bool = True):
        streamer = self.get_response_streamer(user_input)
        generated_response = ""
        for response in streamer:
            generated_response += response
            yield response
        if with_memory:
            self.conversation.append({"role": "assistant", "content": generated_response})

    def clear_history(self):
        self.conversation = []

    def chatbot(self, system_instructions: str = ""):
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


# ---------------------------------------------------------
class PandaChatBot(AbstractChatBot):
    def __init__(
            self,
            model_path: str,
            quantization: Literal["16bit", "8bit", "4bit"] = "16bit",
            setup_prompt: str = "",
            tools=[]
    ):
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

        super().__init__(setup_prompt=setup_prompt, tools=tools)

        def get_response_streamer(
                self, query, max_tokens=1028, temperature=0.6, top_p=0.9
        ):
            self.conversation.append({"role": "user", "content": query})
            # Create tokenized prompt
            prompt = self.tokenizer.apply_chat_template(
                self.conversation,
                tools=self.tools,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
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

    if __name__ == "__main__":
        bot = PandaChatBot(LLAMA_32, quantization="8bit")
        bot.chatbot()
