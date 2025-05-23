from core.Hit import Hit
from abc import ABC, abstractmethod
from typing import Optional

class Resolved:
    """
    Represents a successful resolution
    """
    def __init__(self, hit: Hit, pdf_url: str, source: str) -> None:
        self.hit = hit
        self.pdf_url = pdf_url
        self.source = source

    def __str__(self) -> str:
        return f"Resolved({self.hit.title}, {self.pdf_url}, {self.source})"

    def to_json(self) -> dict:
        """Convert the resolution to a JSON-serializable dictionary."""
        return {
            "title": self.hit.title,
            "doi": self.hit.doi,
            "url": self.hit.url,
            "year": self.hit.year,
            "source": self.hit.source,
            "score": self.hit.score,
            "pdf_url": self.pdf_url,
            "resolver_source": self.source
        }

class BaseResolver(ABC):
    """
    Each resolver gets a 'hit' and returns either 'Resolved' or None representing failure (couldn't be resolved).
    """
    @abstractmethod
    async def resolve(self, hit: Hit) -> Optional[Resolved]:
        pass
