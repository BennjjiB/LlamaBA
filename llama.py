import json
import torch
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
llama_31 = "meta-llama/Llama-3.1-8B-Instruct"  # <-- llama 3.1


with open("tool_definitions.json", "r") as file:
    data = json.load(file) 
    tool_definitions = json.dumps(data)

print(tool_definitions)

setup_prompt = [
    {
        "role": "system",
        "content": f"""
            You are a robot arm named Panda with tool calling capabilities.
            Your task is to grab and sort colored blocks. Respond in a positive manner.
            When you receive a tool call response, use the output to format an answer to the original user question.
            If you are using tools, respond in the format 
            {{"function": function name, "parameters": dictionary of function arguments}}.
            Do not use variables
            {tool_definitions}
        """
    },
    {
        "role": "user",
        "content": """
            Question: what is the weather and traffic looks like in Sydney?
        """
    },
]

pipe = pipeline(model=llama_31, device=device, torch_dtype=torch.bfloat16)
response = pipe(
    setup_prompt,
    do_sample=False,
    temperature=1.0,
    top_p=1,
    max_new_tokens=50
)

print(f"Generation: {response[0]['generated_text']}")

