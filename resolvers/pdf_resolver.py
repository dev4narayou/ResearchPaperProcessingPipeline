from resolvers.base import BaseResolver
from core.Hit import Hit
from resolvers.base import Resolved

class PdfResolver(BaseResolver):
    """
    Basic initial resolver that checks if the hit already has a direct pdf link.
    returns a Resolved object if it does, otherwise returns None.
    """
    def __init__(self):
        super().__init__()

    async def resolve(self, hit: Hit) -> Resolved:
            url = hit.url
            if url.endswith('.pdf'):
                return Resolved(hit=hit, pdf_url=url, source=hit.source)
            else:
                return None


if __name__ == "__main__":
    import asyncio

    async def main():
        hit = Hit(title="test", url="https://doi.org/10.2139/ssrn.5236110", source="test", doi="test", year=2025)
        resolver = PdfResolver()
        resolved = await resolver.resolve(hit)
        print(resolved)

    asyncio.run(main())
