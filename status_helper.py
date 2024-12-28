import queue

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
