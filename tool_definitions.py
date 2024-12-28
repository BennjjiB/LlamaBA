from status_helper import update_status
import json


def calculate(id, args):
    """Evaluate a mathematical expression"""
    update_status(id, "calculate", "The calculator is starting the calculation")
    try:
        result = eval(args.get("expression"))
        update_status(id, "calculate", json.dumps({"result": result}))
    except Exception as e:
        update_status(id, "calculate", json.dumps({"error": f"Unexpected error: {str(e)}"}))


calculate_definition = {
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

tool_definitions = [calculate_definition]

available_tools = {
    "calculate": calculate,
}
