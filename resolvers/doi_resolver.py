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
_EPMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

class DoiResolver(BaseResolver):
    name = "doi"

    async def resolve(self, hit: Hit) -> Optional[Resolved]:
        if not hit.doi:
            return None

        doi = hit.doi.lower()

        # 1. arXiv DOI shortcut
        if doi.startswith("10.48550/arxiv."):
            arxiv_id = doi.split("arxiv.", 1)[-1]
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            return Resolved(hit, pdf_url, "arxiv")

        # 2. Unpaywall
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                res = await client.get(_UNPAYWALL_URL.format(
                    DOI=doi,
                    EMAIL=settings.UNPAYWALL_EMAIL
                ))
                if res.status_code == 200:
                    data = res.json()
                    loc = data.get("best_oa_location") or {}
                    pdf = loc.get("url_for_pdf")
                    if pdf:
                        return Resolved(hit, pdf, "unpaywall")
            except Exception as e:
                print("Unpaywall failed:", e)

        # 3. PubMed Central (Europe PMC)
        epmc_params = {
            "query": f"doi:{doi}",
            "format": "json",
            "pageSize": 1
        }
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                r = await client.get(_EPMC_URL, params=epmc_params)
                r.raise_for_status()
                items = r.json().get("resultList", {}).get("result", [])
                if items:
                    pmcid = items[0].get("pmcid")
                    if pmcid:
                        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf"
                        return Resolved(hit, pdf_url, "pmc")
            except Exception as e:
                print("PMC fallback failed:", e)

        # 4. Domain-specific DOI patterns
        # haven't verified their efficacy
        # ... take what we can get, and move on for now
        if doi.startswith("10.1145/"):  # ACM
            pdf_url = f"https://dl.acm.org/doi/pdf/{doi}"
            return Resolved(hit, pdf_url, "acm")

        if doi.startswith("10.20944/preprints"):  # Preprints.org
            if hit.url and "/manuscript/" in hit.url:
                pdf_url = hit.url.rstrip("/") + "/download/pdf"
                return Resolved(hit, pdf_url, "preprints")

        if doi.startswith("10.31219/osf.io/"):
            osf_id = doi.split("osf.io/", 1)[-1].replace("_v1", "")
            pdf_url = f"https://osf.io/{osf_id}/download"
            return Resolved(hit, pdf_url, "osf")

        if doi.startswith("10.22541/au."):
            pdf_url = f"https://www.authorea.com/doi/pdf/{doi}"
            return Resolved(hit, pdf_url, "authorea")

        # above seem to work so far


        return None

if __name__ == "__main__":
    async def main():
        resolver = DoiResolver()
        hit = Hit(title="", doi="10.3390/nu14050921", url="", year=2022, source="")
        res = await resolver.resolve(hit)
        print(res)

    import asyncio
    asyncio.run(main())