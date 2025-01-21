easy_prompt = [
    "Hi Panda, please sort all bricks by color.",
    "Hi Panda, please sort all bricks by size.",
    "Sort all remaining bricks by their size.",
    "Grab the blue brick.",
    "Hey Panda, there is a red brick on the table. Please grab it.",
    "Could you please sort the yellow brick.",
    "Get all collision-free bricks.",
    "How many bricks can you grab?",
    "How many bricks can you see?",
    "Which bricks can you detect on the table?"
]

easy_target = [
    {"function_name": "sort_all_bricks", "arguments": {"by_color": True}},
    {"function_name": "sort_all_bricks", "arguments": {"by_color": False}},
    {"function_name": "sort_all_bricks", "arguments": {"by_color": False}},
    {"function_name": "grab_brick", "arguments": {"color": "blue"}},
    {"function_name": "grab_brick", "arguments": {"color": "red"}},
    {"function_name": "grab_brick", "arguments": {"color": "yellow"}},
    {"function_name": "get_collision_free_bricks", "arguments": {}},
    {"function_name": "get_collision_free_bricks", "arguments": {}},
    {"function_name": "get_all_bricks", "arguments": {}},
    {"function_name": "get_all_bricks", "arguments": {}},
]

neutral_prompt = [
    "Hi what's your name?",
    "Do you like sorting bricks?",
    "Tell me a joke.",
    "Who build you?",
    "What's your mood today, are you feeling good?",
    "Can you explain me how to make good coffee?",
    "Do you have a favorite tv show?",
    "If you could have any superpower, what would it be and why?",
    "The most important number in the whole universe is ...",
    "Tell me a detailed story based in fact about humanity"
]

hard = [
    "Get me the current weather in Tübingen.",
    "Move your robot arm into the initial position.",
    "Build a house using only red and blue bricks.",
    "Grab the eraser on the right.",
    "Open and close your gripper twice.",
    "Do a cool handshake with me.",
    "Grab the pink brick.",
    "Sort all bricks by their shape.",
    "I really like the color yellow, and today happens to be my birthday. Could you grab a brick as a present for me?",
    "All those bricks are piling up on the table. Could you go and tidy up that mess?"
]

hard_target = [
    False, False, False, False, False, False, False, False,
    {"function_name": "grab_brick", "arguments": {"color": "yellow"}},
    {"function_name": "sort_all_bricks", "arguments": {"by_color": True}}
]
