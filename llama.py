import json
import torch
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
llama_31 = "meta-llama/Llama-3.1-8B-Instruct"  # <-- llama 3.1


def getToolDefinitions(tool_file_path: str):
    with open(tool_file_path, "r") as file:
        data = json.load(file)
        return json.dumps(data)


class Llama3:
    def __init__(self, model_path, tool_path):
        self.model_id = model_path

        self.pipe = pipeline(
            "text-generation",
            model=self.model_id,
            device=device,
            torch_dtype=torch.bfloat16
        )

    def get_response(
        self, query, message_history, max_tokens=1028, temperature=0.6, top_p=0.9
    ):
        user_prompt = message_history + [{"role": "user", "content": query}]
        prompt = self.pipe.tokenizer.apply_chat_template(
            user_prompt, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        response = outputs[0]["generated_text"][len(prompt):]
        return response, user_prompt + [{"role": "assistant", "content": response}]

    def chatbot(self, system_instructions=""):
        conversation = [{"role": "system", "content": system_instructions}]
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the chatbot. Goodbye!")
                break
            response, conversation = self.get_response(
                user_input, conversation)
            print(f"Assistant: {response}")


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
    bot = Llama3(llama_31, "tool_definitions.json")
    bot.chatbot(setup_prompt)
