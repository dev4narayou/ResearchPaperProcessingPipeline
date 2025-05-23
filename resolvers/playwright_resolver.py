from resolvers.base import BaseResolver
from core.Hit import Hit
from resolvers.base import Resolved

class PlaywrightResolver(BaseResolver):
    def __init__(self):
        super().__init__()

    async def resolve(self, hit: Hit) -> Resolved:
        pass