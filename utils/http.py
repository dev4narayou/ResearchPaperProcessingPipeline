import httpx, asyncio, random, logging

async def get(url: str, **params):
    headers = {
        "User-Agent": "ResearchAgent/1.0"
    }
    async with httpx.AsyncClient(timeout=10, headers=headers) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r
