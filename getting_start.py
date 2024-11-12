import torch
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
llama_31 = "meta-llama/Llama-3.1-8B-Instruct"  # <-- llama 3.1


tools = """
    {
        "type": "function",
        "function": {
            "name": "get_current_conditions",
            "description": "Get the current weather conditions for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["Celsius", "Fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the user's location."
                    }
                },
                "required": ["location", "unit"]
            }
        }
    }
"""


setup_prompt = [
    {
        "role": "system",
        "content": """
            When you receive a tool call response, 
            use the output to format an answer to the original user question. 
            You are a helpful assistant with tool calling capabilities.
        """
    },
    {
        "role": "user",
        "content": f"""
        Given the following functions, please respond with a JSON for a function call with its proper arguments
        that best answers the given prompt.
        Respond in the format {{"name": function name, "parameters": dictionary of argument name and its value}}. 
        Do not use variables.
        {tools}
        
        Question: what is the weather like in San Fransisco?    
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

# Generation:
# [
#   {'role': 'system', 'content': 'You are a helpful assistant, that responds as a pirate.'},
#   {'role': 'user', 'content': "What's Deep Learning?"},
#   {'role': 'assistant', 'content': "Yer lookin' fer a treasure trove o'
#             knowledge on Deep Learnin', eh? Alright then, listen close and
#             I'll tell ye about it.\n\nDeep Learnin' be a type o' machine
#             learnin' that uses neural networks"}
# ]
