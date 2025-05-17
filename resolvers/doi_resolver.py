"""
Resolver that tries the cheap DOI-based routes:
• Unpaywall
• arXiv (10.48550/*)
• PubMed Central (Europe PMC full-text)
"""

from resolvers.base import BaseResolver, Resolved
from core.Hit import Hit
from typing import Optional
import httpx
from config.config import settings

_UNPAYWALL_URL = "https://api.unpaywall.org/v2/{DOI}?email={EMAIL}"


class DoiResolver(BaseResolver):

    async def resolve(self, hit: Hit) -> Optional[Resolved]:

        # 1. try Unpaywall API
        if hit.doi:
            async with httpx.AsyncClient() as client:
                res = await client.get(_UNPAYWALL_URL.format(
                    DOI=hit.doi,
                    EMAIL=settings.UNPAYWALL_EMAIL
                ))
                if res.status_code != 200:
                    return None
                data = res.json()
                loc = data.get("best_oa_location") or {}
                pdf = loc.get("url_for_pdf")
                if pdf:
                    return Resolved(hit, pdf, "unpaywall")

        # TODO implement more resolvers


if __name__ == "__main__":
    async def main():
        resolver = DoiResolver()
        hit = Hit(title="", doi="10.3390/nu14050921", url="", year=2022, source="")
        res = await resolver.resolve(hit)
        print(res)

    import asyncio
    asyncio.run(main())




