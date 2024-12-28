import json

import requests

from status_helper import status_queue
from tool_service import ToolService


class Client:
    def __init__(self, base_url: str, transcriber, tool_service: ToolService) -> None:
        self.base_url = base_url
        self.transcriber = transcriber
        self.tool_service = tool_service

    def send_prompt(self, prompt: str, tool_response=False):
        """
        Sends a GET request to the server and processes the response.

        Args:
            base_url (str): The server's base URL.
            prompt (str): The transcription prompt to be processed.
        """
        endpoint = f"{self.base_url}/get-response"
        params = {"prompt": prompt, "is_tool_response": tool_response}

        with requests.get(endpoint, params=params, stream=True, timeout=10) as response:
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                return
            return self.__handle_response(response)

    def __handle_response(self, response):
        generated_response = ""
        for chunk in response.iter_content():
            str = chunk.decode(errors='replace')
            print(str, end='', flush=True)
            generated_response += str
        print("\n")
        tool_threads = self.tool_service.parse_and_execute_response(generated_response)
        if tool_threads:
            self.handle_tool_status_changes(tool_threads)

    def handle_tool_status_changes(self, threads):
        while any(thread.is_alive() for thread in threads) or not status_queue.empty():
            try:
                new_status = status_queue.get(timeout=1)  # Wait for 1 second for an item
            except queue.Empty:
                if not any(thread.is_alive() for thread in threads):
                    print("All threads are finished and the queue is empty. Exiting.")
                    break
                continue
            print(f"Status updated to: {new_status}")

            # Ensure `new_status` contains the necessary keys
            if isinstance(new_status, dict) and all(
                    key in new_status for key in ["tool_call_id", "name", "content"]
            ):
                tool_response = {
                    "tool_call_id": new_status["tool_call_id"],
                    "role": "tool",
                    "name": new_status["name"],
                    "content": new_status["content"],
                }
                self.send_prompt(json.dumps(tool_response), tool_response=True)
            else:
                print("Invalid status update received.")

    def start_chat_interface(self, voice=True):
        if voice:
            while True:
                transcription = self.transcriber.start_transcription(
                    record_timeout=1, phrase_timeout=1.5, speak_timeout=3)
                prompt = "".join(transcription)
                print("\n-------------Generating response-------------\n")
                self.send_prompt(prompt)
        else:
            while True:
                prompt = input('Ask a question: ')
                print("\n-------------Generating response-------------\n")
                self.send_prompt(prompt)


def main():
    # BASE_URL = "http://134.2.17.204:8080"
    BASE_URL = " http://127.0.0.1:5000"
    client = Client(
        BASE_URL,
        None,
        ToolService()
    )
    client.start_chat_interface(voice=False)


if __name__ == "__main__":
    main()
