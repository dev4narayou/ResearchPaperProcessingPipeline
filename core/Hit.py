from dataclasses import dataclass
from typing import Optional

@dataclass
class Hit:
    title: str
    doi: Optional[str]
    url: str # can be a direct pdf link or a link to a page that contains a pdf link (in most cases... can't confirm 100% that all api responses will be like this)
    year: Optional[int]
    source: str
    score: float = 0.0
