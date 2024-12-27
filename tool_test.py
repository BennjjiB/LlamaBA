from groq_bot import GroqChatBot
import threading
import queue
import os
import re
import json

# Shared queue to communicate status changes
status_queue = queue.Queue()


def update_status(id, function_name, new_status):
    """Function to immediately update the status."""
    status_queue.put(
        {
            "tool_call_id": id,
            "name": function_name,
            "content": new_status,
        }
    )


def calculate(id, expression):
    """Evaluate a mathematical expression"""
    update_status(id, "calculate", "The calculator is starting the calculation")
    try:
        result = eval(expression)
        update_status(id, "calculate", json.dumps({"result": result}))
    except Exception as e:
        update_status(id, "calculate", json.dumps({"error": f"Unexpected error: {str(e)}"}))
    finally:
        update_status(id, "calculate", "STOP")


tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]


def chat_with_tool_calls():
    """Function to monitor status changes."""
    setup_prompt = "You are a calculator assistant. Use the calculate function to perform mathematical operations and provide the results."
    bot = GroqChatBot(setup_prompt=setup_prompt, tools=tools)
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chatbot. Goodbye!")
            break
        generated_response = ""

        tool_calls = None
        for response in bot.generate_chat_response(user_input):
            print(response)
            if isinstance(response, str):
                generated_response += response
            else:
                tool_calls = response

        if tool_calls:
            # Define the available tools that can be called by the LLM
            available_functions = {
                "calculate": calculate,
            }
            # Process each tool call
            threads = []
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                thread = threading.Thread(target=function_to_call,
                                          args=(tool_call.id, function_args.get("expression")))
                threads.append(thread)
                thread.start()

            while any(thread.is_alive() for thread in threads) or status_queue.not_empty:
                new_status = status_queue.get()  # Blocks until an item is available
                if new_status == "STOP":
                    print("Monitor received stop signal. Exiting.")
                    break
                print(f"Status updated to: {new_status}")
                tool_repsonse = {
                    "tool_call_id": new_status["tool_call_id"],
                    "role": "tool",
                    "name": new_status["name"],
                    "content": new_status["content"],
                }

                for response in bot.generate_chat_response(tool_repsonse, tool_response=True):
                    print(response)


chat_with_tool_calls()

# matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", generated_response, re.DOTALL)
#         tool_calls = []
#         for match in matches:
#             try:
#                 tool_calls.append(json.loads(match))
#             except json.JSONDecodeError:
#                 print(f"Invalid JSON content: {match}")
