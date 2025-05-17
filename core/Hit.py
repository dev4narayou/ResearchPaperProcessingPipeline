from dataclasses import dataclass
from typing import Optional

@dataclass
class Hit:
    title: str
    doi: Optional[str]
    url: str
    year: Optional[int]
    source: str
    score: float = 0.0
