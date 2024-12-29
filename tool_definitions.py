from status_helper import update_status
import json


def calculate(args):
    """
    Evaluate a mathematical expression

     Args:
        id (str): Identifier for the calculation process.
        args (dict): Contains "expression" (str), the math expression to evaluate.
    """
    id = args.get("tool_id", None)
    update_status("calculate", "Calculator is booting...", id)
    try:
        result = eval(args.get("expression"))
        update_status("calculate", json.dumps({"result": result}), id)
    except Exception as e:
        update_status("calculate", json.dumps({"error": f"Unexpected error: {str(e)}"}), id)


calculate_definition = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Note the calculator first has to boot!",
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
