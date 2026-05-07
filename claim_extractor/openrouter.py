"""
Shared OpenRouter API client for all pipeline phases.

Usage:
    from openrouter import OpenRouterClient

    client = OpenRouterClient(api_key, model)
    client.semaphore = asyncio.Semaphore(workers)

    response, error = await client.complete(session, system_prompt, user_prompt)
"""

import asyncio
import aiohttp
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.semaphore = None

    async def complete(
        self,
        session: aiohttp.ClientSession,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.3,
        retries: int = 3,
        timeout: int = 120,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Send a chat completion request to OpenRouter.

        Returns:
            (response_text, error_string) — one will be None.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/crossvalqa",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        last_error = None
        for attempt in range(retries):
            try:
                async with self.semaphore:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data["choices"][0]["message"]["content"], None
                        elif resp.status == 429:
                            wait_time = 5 * (attempt + 1)
                            logger.warning(f"Rate limited on {self.model}, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            last_error = "Rate limited"
                        elif 500 <= resp.status < 600:
                            wait_time = 2 * (attempt + 1)
                            text = await resp.text()
                            last_error = f"HTTP {resp.status}: {text[:200]}"
                            logger.warning(f"Server error {resp.status}, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                        else:
                            text = await resp.text()
                            last_error = f"HTTP {resp.status}: {text[:200]}"
                            return None, last_error  # 4xx (not 429) — don't retry

            except asyncio.TimeoutError:
                last_error = f"Timeout ({timeout}s)"
                logger.warning(f"Timeout on attempt {attempt + 1}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Error on attempt {attempt + 1}: {e}")

            await asyncio.sleep(2 * (attempt + 1))

        return None, last_error
