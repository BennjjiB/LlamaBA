import requests
from transcriber import Transcriber
from tool_definitions import ToolService


class Client:
    def __init__(self, base_url: str, transcriber, tool_parser: ToolService) -> None:
        self.base_url = base_url
        self.transcriber = transcriber
        self.tool_parser = tool_parser

    def send_prompt(self, prompt: str):
        """
        Sends a GET request to the server and processes the response.

        Args:
            base_url (str): The server's base URL.
            prompt (str): The transcription prompt to be processed.
        """
        endpoint = f"{self.base_url}/get-response"
        params = {"prompt": prompt}

        with requests.get(endpoint, params=params, stream=True, timeout=10) as response:
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                return
            return self.__handle_response(response)

    def __handle_response(self, response):
        tool_result = self.tool_parser.parse_and_execute_response(
            response.iter_content(chunk_size=1024))
        # Can't parse response to tool calls so just display the message
        if isinstance(tool_result, tuple):
            print(tool_result[0], end='')
            for chunk in tool_result[1]:
                print(chunk.decode('utf-8', errors='replace'), end='', flush=True)
            print("\n")

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


def get_current_time():
    return 42


def main():
    client = Client(
        "http://127.0.0.1:8000",
        None,
        ToolService({"current_time": get_current_time})
    )
    client.start_chat_interface(voice=False)


if __name__ == "__main__":
    main()
