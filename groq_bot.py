import os
import json
from groq import Groq
from sympy.polys.polyconfig import query

from chatbot import AbstractChatBot


class GroqChatBot(AbstractChatBot):
    def __init__(self, model="llama3-groq-70b-8192-tool-use-preview", setup_prompt="", tools=[]):
        self.client = Groq(
            api_key="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",
        )
        self.model = model
        super().__init__(setup_prompt=setup_prompt, tools=tools)

    def get_response_streamer(
            self, query, max_tokens=4096, temperature=0.6, top_p=0.9
    ):
        self.conversation.append(query)
        response = self.client.chat.completions.create(
            messages=self.conversation,
            model=self.model,
            tools=self.tools,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=None,
            tool_choice="auto",
            stream=False,
        )
        return response.choices[0].message

    def generate_chat_response(self, user_input: str, tool_response: bool = False, with_memory: bool = True):
        # change to role to tool ...
        query = {"role": "user", "content": user_input} if tool_response else {"role": "user",
                                                                               "content": user_input}
        response_message = self.get_response_streamer(query)
        tool_calls = response_message.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                tool = {
                    "id": tool_call.id,
                    "function_name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }
                tool_string = json.dumps(tool)
                yield tool_string
                self.conversation.append({"role": "assistant", "content": tool_string})
        else:
            yield response_message.content
            self.conversation.append(
                {"role": "assistant", "content": response_message.content}
            )
