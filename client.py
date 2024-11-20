import requests
from Transcriber import Transcriber


def send_prompt(base_url: str, prompt: str):
    """
    Sends a GET request to the server and processes the response.

    Args:
        base_url (str): The server's base URL.
        prompt (str): The transcription prompt to be processed.
    """
    endpoint = f"{base_url}/get-response"
    params = {"prompt": prompt}

    with requests.get(endpoint, params=params, stream=True, timeout=10) as response:
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            return

        # In the case of SSE, we handle the stream by reading it line by line.
        for line in response.iter_lines():
            print(f"{line.decode('utf-8').strip()}")


def main():
    base_url = "http://127.0.0.1:8000"
    transcriber = Transcriber()

    while True:
        transcription = transcriber.start_transcription()
        prompt = "".join(transcription)
        print("------------------------\n")
        print("Sending: ", prompt)
        print("------------------------\n")
        send_prompt(base_url, prompt)


if __name__ == "__main__":
    main()
