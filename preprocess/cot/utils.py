import json
from typing import List, Dict
import os
from dotenv import load_dotenv
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

class ChatLLM():
    def __init__(self, base_url, key, prompt, model, temperature, max_tokens=None):
        self.base_url = base_url
        if type(key) is tuple:
            self.key = key[0]
        elif type(key) is str:
            self.key = key
        else:
            raise Exception('key must be a string or a tuple')
        self.prompt = prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def _async_chat(self, session, req: Dict):
        prompt = self.prompt.format(**req)
        # if req has image_Url
        images = req.get('images', None)
        content = [
            {"type": "text", "text": prompt}
        ]
        if images:
            for image in images:
                content.append(
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image}'}}
                )
        data = json.dumps({
            'messages': [{'role': 'user', 'content': content}],
            'model': self.model,
            'stream': False,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        })
        
        for attempt in range(3):
            response_json = None
            try:
                async with session.post(
                    url=self.base_url,
                    headers={
                        'Authorization': f'Bearer {self.key}',
                        'Content-Type': 'application/json'
                        },
                    data=data
                ) as response:
                    response_json = await response.json()
                    return response_json['choices'][0]['message']['content']
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(30)
                else:
                    if response_json is None:
                        raise Exception(e)
                    else:
                        raise Exception(response_json)

    async def _async_chat_batch(self, reqs: List[Dict]):
        async with aiohttp.ClientSession() as session:
            tasks = [self._async_chat(session, request) for request in reqs]
            return await asyncio.gather(*tasks)

    def chat_batch(self, requests: List[Dict]):
        with ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: asyncio.run(self._async_chat_batch(requests)))
            return future.result()
    