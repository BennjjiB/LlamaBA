import aiohttp
import asyncio
from Transcriber import Transcriber


async def send_prompt(session: aiohttp.ClientSession, base_url: str, prompt: str):
    """
    Makes a GET request to the endpoint /get-response.

    Args:
        session (aiohttp.ClientSession): A session to manage HTTP connections.
        base_url (str): The base URL of the server.
        prompt (str): The prompt for which the LLM should respond.
    """
    endpoint = f"{base_url}/get-response"
    params = {"prompt": prompt}

    try:
        async with session.get(endpoint, params=params) as response:
            if response.status != 200:
                print(f"Error: Received status code {response.status}")
                return
            print("Connected to SSE stream. Receiving events:")
            async for line in response.content.iter_any():
                line = line.decode("utf-8").strip()
                if line:
                    print(f"Event: {line}")
    except aiohttp.ClientError as e:
        print(f"An error occurred while connecting to the server: {e}")


async def main():
    base_url = "http://127.0.0.1:8000"
    transcriber = Transcriber()

    tasks = []  # Store all ongoing tasks for sending prompts
    async with aiohttp.ClientSession() as session:
        # Start the transcription and process prompts
        async for transcription in transcriber.start_transcription():
            prompt = "".join(transcription)
            task = asyncio.create_task(send_prompt(session, base_url, prompt))
            tasks.append(task)
        await asyncio.gather(*tasks)
        
if __name__ == "__main__":
    asyncio.run(main())
