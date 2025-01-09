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

sort_bricks_definition = {
    "type": "function",
    "function": {
        "name": "sort_bricks",
        "description": "Sorts all bricks by color.",
        "parameters": {},
    },
}

tool_definitions = [calculate_definition, sort_bricks_definition]
available_tools = []