import httpx, asyncio, random, logging
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

async def get(url: str, **params):
    headers = {
        "User-Agent": "ResearchAgent/1.0"
    }

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    ):
        try:
            async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                return r
        except Exception as e:
            if attempt.retry_state.attempt_number == 3:  # Last attempt
                raise
            logging.warning(f"Request failed, retrying: {str(e)}")
            continue
