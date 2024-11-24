import json
from typing import Callable


class ToolService():
    def __init__(self, tool_definitions: dict[str, Callable]):
        self.tool_definitions = tool_definitions

    def parse_and_execute_response(self, stream):
        calls = self.__parse_tool_response(stream)
        if not isinstance(calls, list):
            return calls
        for result in self.__execute_tool_calls(calls):
            print(result)

    def __parse_tool_response(self, stream):
        response_string = ""
        for chunk in stream:
            response_string += chunk.decode('utf-8', errors='replace').strip()
            if not (response_string.startswith("[") or response_string.startswith("{")):
                return response_string, stream
        print(response_string)
        parsed_response = json.loads(response_string)
        if isinstance(parsed_response, list):
            return parsed_response
        return [parsed_response]

    def __execute_tool_calls(self, calls):
        for call in calls:
            print(call)
            name = call["function"]
            params = call["parameters"]
            tool = self.tool_definitions[name]
            if tool:
                print("executing", name)
                yield tool(**params)


current_time = {
    "type": "function",
    "function": {
        "name": "current_time",
        "description": "Get the current local time as a string.",
        "parameters": {
            'type': 'object',
            'properties': {}
        }
    }
}

# A more complete function that takes two numerical arguments
multiply = {
    'type': 'function',
    'function': {
        'name': 'multiply',
        'description': 'A function that multiplies two numbers',
        'parameters': {
            'type': 'object',
            'properties': {
                'a': {
                    'type': 'number',
                    'description': 'The first number to multiply'
                },
                'b': {
                    'type': 'number', 'description': 'The second number to multiply'
                }
            },
            'required': ['a', 'b']
        }
    }
}

tools = [current_time]
