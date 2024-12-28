import json
from typing import Callable
from typing import Dict
import threading

from tool_definitions import available_tools
import re


class ToolService():
    def __init__(self, available_tools=available_tools):
        self.available_tools = available_tools

    def parse_and_execute_response(self, tools):
        parsed_tools = self.parse_tools(tools)
        threads = []
        if parsed_tools:
            for tool in parsed_tools:
                tool_call = json.loads(tool)
                # Works only for groq may change later
                tool_id = tool_call["id"]
                function_name = tool_call["function_name"]
                function_to_call = self.available_tools[function_name]
                function_args = json.loads(tool_call["arguments"])
                threads.append(
                    self.start_tool_call(tool_id, function_to_call, function_args)
                )
        return threads

    def start_tool_call(self, id, function_to_call, function_args):
        thread = threading.Thread(target=function_to_call, args=(id, function_args))
        thread.start()
        return thread

    def parse_tools(self, tools):
        return parse_groq(tools)


def parse_groq(text):
    """
    Extract substrings with outermost curly braces {}.
    Handles nested braces correctly.
    """
    result = []
    stack = []
    current = []
    for char in text:
        if char == '{':
            if stack:
                current.append(char)
            stack.append('{')
        elif char == '}':
            stack.pop()
            if stack:
                current.append(char)
            else:
                result.append('{' + ''.join(current) + '}')
                current = []
        elif stack:
            current.append(char)
    return result
