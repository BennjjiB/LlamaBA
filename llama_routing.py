from chatbot import AbstractChatBot
from groq import Groq
import json

# Initialize the Groq client
client = Groq(api_key="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")


def calculate(expression):
    """Tool to evaluate a mathematical expression"""
    try:
        result = eval(expression)
        return json.dumps({"result": result})
    except:
        return json.dumps({"error": "Invalid expression"})


# Routing setup
#  "You are a routing assistant. Determine if tools are needed based on the user query."
#         max_tokens=20  # We only need a short response

class Router:
    def __init__(self,
                 routing_model: AbstractChatBot,
                 gneral_model: AbstractChatBot,
                 tool_model: AbstractChatBot
                 ):
        self.routing_model = routing_model
        self.general_model = general_model
        self.tool_use_model = tool_use_model

    def route_query(query):
        """Routing logic to let LLM decide if tools are needed"""
        routing_prompt = f"""
            Given the following user query, determine if any tools are needed to answer it.
            If a calculation tool is needed, respond with 'TOOL: ROBOT'.
            If no tools are needed, respond with 'NO TOOL'.
        
            User query: {query}
        
            Response:
            """

        response = "".join(
            self.routing_model.generate_chat_response(routing_prompt)
        )
        routing_decision = response.strip()

        if "TOOL: ROBOT" in routing_decision:
            return "robot"
        else:
            return "no robot needed"

    def process_query(query):
        """Process the query and route it to the appropriate model"""
        route = self.route_query(query)
        if route == "robot":
            response = run_with_tool(query)
        else:
            response = run_general(query)

        return {
            "query": query,
            "route": route,
            "response": response
        }

    def run_with_tool(query):
        """Use the tool use model to perform the calculation"""
        response = self.tool_use_model.generate_chat_response(query)
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        if tool_calls:
            messages.append(response_message)
            for tool_call in tool_calls:
                function_args = json.loads(tool_call.function.arguments)
                function_response = calculate(function_args.get("expression"))
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": "calculate",
                        "content": function_response,
                    }
                )
            second_response = client.chat.completions.create(
                model=TOOL_USE_MODEL,
                messages=messages
            )
            return second_response.choices[0].message.content
        return response_message.content


def run_general(query):
    """Use the general model to answer the query since no tool is needed"""
    response = client.chat.completions.create(
        model=GENERAL_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content


# Example usage
if __name__ == "__main__":
    queries = [
        "What is the capital of the Netherlands?",
        "Calculate 25 * 4 + 10"
    ]

    for query in queries:
        result = process_query(query)
        print(f"Query: {result['query']}")
        print(f"Route: {result['route']}")
        print(f"Response: {result['response']}\n")
