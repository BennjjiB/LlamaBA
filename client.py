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
        for chunk in response.iter_content(chunk_size=1024):
            print(chunk.decode('utf-8', errors='replace'), end='', flush=True)
        print("\n")

def main():
    base_url = "http://127.0.0.1:8000"
    transcriber = Transcriber()

    while True:
        transcription = transcriber.start_transcription(record_timeout=1, phrase_timeout=1.5, speak_timeout=3)
        prompt = "".join(transcription)
        print("\n-------------Generating response-------------\n")
        send_prompt(base_url, prompt)


if __name__ == "__main__":
    main()
