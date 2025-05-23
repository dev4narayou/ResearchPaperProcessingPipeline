from generators.keyword_expander import topic_to_queries
from searchers.meta_searcher import MetaSearcher
from typing import List, Optional
from core.Hit import Hit
import asyncio
import json
from dataclasses import asdict, dataclass
from resolvers import ResolverManager
from resolvers.base import Resolved
import logging
from pathlib import Path
import hashlib
import aiohttp
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential
import asyncpg
from openai import AsyncOpenAI
import itertools
from typing import Dict, Any
from config.config import settings

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# data classes for typed payloads
@dataclass
class Downloaded:
    resolved: Resolved
    pdf: bytes
    sha256: str

@dataclass
class Parsed:
    resolved: Resolved
    text: str
    metadata: Dict[str, Any]
    sha256: str

@dataclass
class Summarized:
    resolved: Resolved
    summaries: Dict[str, str]
    embeddings: Dict[str, List[float]]
    sha256: str

# global rate limiters
DOWNLOAD_SEMAPHORE = asyncio.Semaphore(settings.MAX_CONCURRENT_DOWNLOADS)
LLM_SEMAPHORE = asyncio.Semaphore(settings.MAX_CONCURRENT_LLM_CALLS)

# add this before the worker definitions
class DeduplicationTracker:
    def __init__(self):
        self.processed_urls = set()
        self.processed_sha256s = set()
        self._lock = asyncio.Lock()

    async def is_url_processed(self, url: str) -> bool:
        async with self._lock:
            return url in self.processed_urls

    async def mark_url_processed(self, url: str):
        async with self._lock:
            self.processed_urls.add(url)

    async def is_sha256_processed(self, sha256: str) -> bool:
        async with self._lock:
            return sha256 in self.processed_sha256s

    async def mark_sha256_processed(self, sha256: str):
        async with self._lock:
            self.processed_sha256s.add(sha256)

async def download_worker(in_q: asyncio.Queue,
                         out_q: asyncio.Queue,
                         session: aiohttp.ClientSession,
                         dedup: DeduplicationTracker):
    while True:
        try:
            resolved: Resolved = await in_q.get()
            if resolved is None:  # Poison pill
                break

            # Skip if already processed
            if await dedup.is_url_processed(resolved.pdf_url):
                in_q.task_done()
                continue

            try:
                async with DOWNLOAD_SEMAPHORE:
                    async for attempt in AsyncRetrying(
                        stop=stop_after_attempt(settings.MAX_RETRIES),
                        wait=wait_exponential(
                            multiplier=1,
                            min=settings.RETRY_MIN_WAIT,
                            max=settings.RETRY_MAX_WAIT
                        )
                    ):
                        try:
                            # Base headers
                            headers = {
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                                "Accept": "application/pdf,application/x-pdf,application/octet-stream,*/*",
                                "Accept-Language": "en-US,en;q=0.5",
                                "Accept-Encoding": "gzip, deflate, br",
                                "Connection": "keep-alive",
                                "Upgrade-Insecure-Requests": "1",
                                "Sec-Fetch-Dest": "document",
                                "Sec-Fetch-Mode": "navigate",
                                "Sec-Fetch-Site": "none",
                                "Sec-Fetch-User": "?1",
                                "Cache-Control": "max-age=0"
                            }

                            # Add specific headers for PMC URLs
                            if "ncbi.nlm.nih.gov" in resolved.pdf_url:
                                headers.update({
                                    "Referer": "https://www.ncbi.nlm.nih.gov/",
                                    "Accept": "application/pdf,application/x-pdf,application/octet-stream,*/*",
                                    "DNT": "1"
                                })
                                # Try alternative PMC URL format if the original fails
                                if "/pdf" in resolved.pdf_url:
                                    alt_url = resolved.pdf_url.replace("/pdf", "/pdf/")
                                    try:
                                        async with session.get(alt_url, headers=headers) as response:
                                            if response.status == 200:
                                                pdf_bytes = await response.read()
                                                sha256 = hashlib.sha256(pdf_bytes).hexdigest()
                                                await out_q.put(Downloaded(
                                                    resolved=resolved,
                                                    pdf=pdf_bytes,
                                                    sha256=sha256
                                                ))
                                                await dedup.mark_url_processed(resolved.pdf_url)
                                                logger.info(f"Downloaded {alt_url}")
                                                break
                                    except Exception as e:
                                        logger.warning(f"Alternative PMC URL failed: {str(e)}")

                            async with session.get(resolved.pdf_url, headers=headers) as response:
                                if response.status == 200:
                                    pdf_bytes = await response.read()
                                    sha256 = hashlib.sha256(pdf_bytes).hexdigest()
                                    await out_q.put(Downloaded(
                                        resolved=resolved,
                                        pdf=pdf_bytes,
                                        sha256=sha256
                                    ))
                                    await dedup.mark_url_processed(resolved.pdf_url)
                                    logger.info(f"Downloaded {resolved.pdf_url}")
                                    break
                                else:
                                    raise Exception(f"HTTP {response.status}")
                        except Exception as e:
                            logger.error(f"Download failed for {resolved.pdf_url}: {str(e)}")
                            raise
            except Exception as e:
                logger.error(f"Failed to process {resolved.pdf_url} after all retries: {str(e)}")
        except Exception as e:
            logger.exception("Worker error")
        finally:
            in_q.task_done()

async def process_worker(in_q: asyncio.Queue,
                        out_q: asyncio.Queue,
                        loop: asyncio.AbstractEventLoop,
                        dedup: DeduplicationTracker):
    while True:
        try:
            downloaded: Downloaded = await in_q.get()
            if downloaded is None:  # Poison pill
                break

            # Process all downloaded PDFs
            try:
                # Run CPU-intensive work in thread pool
                text = await loop.run_in_executor(
                    None,
                    extract_text,
                    downloaded.pdf
                )

                metadata = await loop.run_in_executor(
                    None,
                    extract_metadata,
                    text
                )

                await out_q.put(Parsed(
                    resolved=downloaded.resolved,
                    text=text,
                    metadata=metadata,
                    sha256=downloaded.sha256
                ))
                # Don't mark as processed here - let it flow through the pipeline
                logger.info(f"Processed {downloaded.resolved.pdf_url}")
            except Exception as e:
                logger.error(f"Failed to process {downloaded.resolved.pdf_url}: {str(e)}")
        except Exception as e:
            logger.exception("Worker error")
        finally:
            in_q.task_done()

async def summarize_worker(in_q: asyncio.Queue,
                          out_q: asyncio.Queue,
                          llm: AsyncOpenAI,
                          dedup: DeduplicationTracker):
    while True:
        try:
            parsed: Parsed = await in_q.get()
            if parsed is None:  # Poison pill
                break

            # Process all items that reach this stage
            try:
                async with LLM_SEMAPHORE:
                    summaries = await generate_summaries(llm, parsed.text)
                    embeddings = await generate_embeddings(llm, parsed.text)

                    await out_q.put(Summarized(
                        resolved=parsed.resolved,
                        summaries=summaries,
                        embeddings=embeddings,
                        sha256=parsed.sha256
                    ))
                    # Don't mark as processed here - only after database storage
                    logger.info(f"Summarized {parsed.resolved.pdf_url}")
            except Exception as e:
                logger.error(f"Failed to summarize {parsed.resolved.pdf_url}: {str(e)}")
        except Exception as e:
            logger.exception("Worker error")
        finally:
            in_q.task_done()

async def store_worker(in_q: asyncio.Queue,
                      pool: asyncpg.Pool,
                      dedup: DeduplicationTracker):
    """Store processed papers with batch operations."""
    batch_size = 3  # Reduced from 10 to process smaller batches
    batch = []
    items_received = 0

    while True:
        try:
            summarized: Summarized = await in_q.get()
            if summarized is None:  # Poison pill
                logger.info(f"Store worker received poison pill. Processing final batch of {len(batch)} items.")
                # Process any remaining items in batch
                if batch:
                    await process_batch(batch, pool, dedup)
                break

            items_received += 1
            logger.info(f"Store worker received item #{items_received}: {summarized.resolved.pdf_url}")

            # Add all items to batch for processing
            batch.append(summarized)

            # Process batch if it reaches the size limit
            if len(batch) >= batch_size:
                logger.info(f"Batch size reached ({len(batch)}), processing batch...")
                await process_batch(batch, pool, dedup)
                batch = []

        except Exception as e:
            logger.exception(f"Store worker error: {e}")
        finally:
            in_q.task_done()

async def process_batch(batch: List[Summarized], pool: asyncpg.Pool, dedup: DeduplicationTracker):
    """Process a batch of summarized papers."""
    try:
        logger.info(f"Starting to process batch of {len(batch)} papers")

        # Test database connection first
        try:
            async with pool.acquire() as conn:
                logger.info("Successfully acquired database connection")

                async with conn.transaction():
                    logger.info("Started database transaction")

                    # Prepare batch data
                    paper_data = []
                    summary_data = []
                    embedding_data = []

                    for i, summarized in enumerate(batch):
                        logger.info(f"Processing paper {i+1}/{len(batch)}: {summarized.resolved.hit.doi}")

                        # Paper data
                        paper_data.append((
                            summarized.resolved.hit.doi,
                            summarized.resolved.hit.title,
                            summarized.resolved.hit.year,
                            summarized.resolved.pdf_url,
                            summarized.sha256,
                            summarized.resolved.source
                        ))

                        # Summary data
                        for summary_type, content in summarized.summaries.items():
                            logger.info(f"Adding {summary_type} summary for {summarized.resolved.hit.doi}")
                            summary_data.append((
                                summarized.resolved.hit.doi,
                                summary_type,
                                content
                            ))

                        # Embedding data
                        for embedding_type, embedding in summarized.embeddings.items():
                            logger.info(f"Adding {embedding_type} embedding for {summarized.resolved.hit.doi}")
                            embedding_json = json.dumps(embedding)
                            embedding_data.append((
                                summarized.resolved.hit.doi,
                                embedding_type,
                                embedding_json
                            ))

                    # Batch insert papers
                    logger.info(f"Inserting {len(paper_data)} papers into database")
                    await conn.executemany("""
                        INSERT INTO papers (doi, title, year, pdf_url, sha256, source)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (doi) DO UPDATE SET
                            title = EXCLUDED.title,
                            year = EXCLUDED.year,
                            pdf_url = EXCLUDED.pdf_url,
                            sha256 = EXCLUDED.sha256,
                            source = EXCLUDED.source
                    """, paper_data)
                    logger.info("Papers inserted successfully")

                    # Batch insert summaries
                    logger.info(f"Inserting {len(summary_data)} summaries into database")
                    await conn.executemany("""
                        INSERT INTO summaries (paper_id, summary_type, content)
                        SELECT id, $2, $3
                        FROM papers
                        WHERE doi = $1
                        ON CONFLICT (paper_id, summary_type) DO UPDATE SET
                            content = EXCLUDED.content
                    """, summary_data)
                    logger.info("Summaries inserted successfully")

                    # Batch insert embeddings
                    logger.info(f"Inserting {len(embedding_data)} embeddings into database")
                    await conn.executemany("""
                        INSERT INTO embeddings (paper_id, embedding_type, embedding)
                        SELECT id, $2, $3
                        FROM papers
                        WHERE doi = $1
                        ON CONFLICT (paper_id, embedding_type) DO UPDATE SET
                            embedding = EXCLUDED.embedding
                    """, embedding_data)
                    logger.info("Embeddings inserted successfully")

                    # Mark all items as processed
                    for summarized in batch:
                        await dedup.mark_sha256_processed(summarized.sha256)
                        logger.info(f"Successfully stored all data for {summarized.resolved.pdf_url}")

                    logger.info("Database transaction completed successfully")

        except Exception as db_error:
            logger.error(f"Database operation failed: {str(db_error)}")
            logger.exception("Database error details:")
            raise

        logger.info("Batch processing completed successfully")

    except Exception as e:
        logger.error(f"Failed to process batch: {str(e)}")
        logger.exception("Full traceback:")
        raise

async def run_pipeline(resolved_hits: List[Resolved]):
    # create queues with backpressure
    download_q = asyncio.Queue(maxsize=settings.QUEUE_MAX_SIZE)
    parse_q = asyncio.Queue(maxsize=settings.QUEUE_MAX_SIZE)
    summarize_q = asyncio.Queue(maxsize=settings.QUEUE_MAX_SIZE)
    store_q = asyncio.Queue(maxsize=settings.QUEUE_MAX_SIZE)

    # Create shared deduplication tracker
    dedup = DeduplicationTracker()

    # Track active tasks
    active_tasks = set()

    # create persistent resources
    async with aiohttp.ClientSession() as session, \
               AsyncOpenAI(api_key=settings.OPENAI_API_KEY) as llm, \
               asyncpg.create_pool(dsn=settings.DATABASE_URL) as pool:

        # get event loop for CPU work
        loop = asyncio.get_running_loop()

        # Start workers
        workers = [
            asyncio.create_task(download_worker(download_q, parse_q, session, dedup)),
            asyncio.create_task(process_worker(parse_q, summarize_q, loop, dedup)),
            asyncio.create_task(summarize_worker(summarize_q, store_q, llm, dedup)),
            asyncio.create_task(store_worker(store_q, pool, dedup))
        ]

        try:
            # queue initial work
            for resolved in resolved_hits:
                # Check if URL is already processed before queuing
                if not await dedup.is_url_processed(resolved.pdf_url):
                    await download_q.put(resolved)
                    active_tasks.add(resolved.pdf_url)

            # wait for queues to empty
            await asyncio.gather(
                download_q.join(),
                parse_q.join(),
                summarize_q.join(),
                store_q.join()
            )
        finally:
            # send poison pills
            for q in (download_q, parse_q, summarize_q, store_q):
                await q.put(None)

            # wait for workers to finish
            await asyncio.gather(*workers, return_exceptions=True)

            # Log final stats
            logger.info(f"Pipeline completed. Processed {len(active_tasks)} unique PDFs.")

# helper functions (implement these based on your needs)
def extract_text(pdf_bytes: bytes) -> str:
    """extract text from pdf with proper error handling."""
    try:
        import io
        import PyPDF2

        # create a bytesio object from the pdf bytes
        pdf_file = io.BytesIO(pdf_bytes)

        # create a pdf reader
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # extract text from all pages
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            text += page_text + "\n"

        # clean up the text (remove extra whitespace, etc.)
        text = " ".join(text.split())

        if not text.strip():
            raise Exception("No text could be extracted from the PDF")

        logger.info(f"Extracted {len(text)} characters of text from PDF with {len(pdf_reader.pages)} pages")
        return text

    except ImportError:
        logger.error("PyPDF2 not installed. Please install it with: pip install PyPDF2")
        raise Exception("PyPDF2 not installed")
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {str(e)}")
        raise

def extract_metadata(text: str) -> Dict[str, Any]:
    """extract structured metadata from text."""
    try:
        # todo: implement actual metadata extraction
        # for now, return simulated metadata
        return {
            "keywords": ["creatine", "exercise", "performance"],
            "abstract": "A simulated abstract for testing.",
            "citations": 42,
            "publication_date": "2023-01-01",
            "journal": "Journal of Sports Science",
            "authors": ["John Doe", "Jane Smith"],
            "institutions": ["University of Sports"],
            "methodology": "Randomized controlled trial",
            "sample_size": 100,
            "duration": "12 weeks"
        }
    except Exception as e:
        logger.error(f"Failed to extract metadata: {str(e)}")
        raise

async def generate_summaries(llm: AsyncOpenAI, text: str) -> Dict[str, str]:
    """generate comprehensive summaries using llm."""
    try:
        # split text into chunks if too long
        max_chunk_size = 4000  # adjust based on model context window
        chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

        logger.info(f"Generating summaries for text of length {len(text)} chars, split into {len(chunks)} chunks")
        summaries = {}

        # generate executive summary
        logger.info("Generating executive summary...")
        executive_prompt = f"Provide a concise executive summary of the following research paper:\n\n{chunks[0]}"
        executive_response = await llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": executive_prompt}],
            max_tokens=500
        )
        summaries["executive"] = executive_response.choices[0].message.content
        logger.info(f"Executive summary generated: {summaries['executive'][:100]}...")

        # generate key findings
        logger.info("Generating key findings...")
        findings_prompt = f"List the key findings and conclusions from this research paper:\n\n{chunks[0]}"
        findings_response = await llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": findings_prompt}],
            max_tokens=500
        )
        summaries["key_findings"] = findings_response.choices[0].message.content
        logger.info(f"Key findings generated: {summaries['key_findings'][:100]}...")

        # generate methodology summary
        logger.info("Generating methodology summary...")
        methodology_prompt = f"Describe the methodology and experimental design of this research paper:\n\n{chunks[0]}"
        methodology_response = await llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": methodology_prompt}],
            max_tokens=500
        )
        summaries["methodology"] = methodology_response.choices[0].message.content
        logger.info(f"Methodology summary generated: {summaries['methodology'][:100]}...")

        # generate implications
        logger.info("Generating implications...")
        implications_prompt = f"Discuss the implications and potential applications of this research:\n\n{chunks[0]}"
        implications_response = await llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": implications_prompt}],
            max_tokens=500
        )
        summaries["implications"] = implications_response.choices[0].message.content
        logger.info(f"Implications generated: {summaries['implications'][:100]}...")

        logger.info("All summaries generated successfully")
        return summaries
    except Exception as e:
        logger.error(f"Failed to generate summaries: {str(e)}")
        logger.exception("Full traceback:")
        raise

async def generate_embeddings(llm: AsyncOpenAI, text: str) -> Dict[str, List[float]]:
    """generate embeddings for different sections of the text."""
    try:
        # truncate text sections to fit within token limits
        # rough estimate: 1 token ≈ 4 characters, 8192 tokens ≈ 32,768 chars
        max_chars = 30000  # conservative limit to stay under 8192 tokens

        sections = {
            "title": text[:200],  # first 200 chars for title
            "abstract": text[200:1000],  # next 800 chars for abstract
            "full_text": text[:max_chars] if len(text) > max_chars else text  # truncated full text
        }

        embeddings = {}
        for section_name, section_text in sections.items():
            if len(section_text.strip()) == 0:
                # skip empty sections
                continue

            try:
                response = await llm.embeddings.create(
                    model="text-embedding-3-small",
                    input=section_text
                )
                embeddings[section_name] = response.data[0].embedding
                logger.info(f"Generated {section_name} embedding ({len(section_text)} chars)")
            except Exception as e:
                logger.warning(f"Failed to generate {section_name} embedding: {str(e)}")
                # continue with other sections instead of failing completely

        if not embeddings:
            raise Exception("Failed to generate any embeddings")

        return embeddings
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        raise

# main execution
if __name__ == "__main__":
    # 1. get a initial topic
    topic = "creatine for exercise"
    # 2. expand the topic into a list of queries
    queries: List[str] = topic_to_queries(topic)
    # 3. search for the queries w/ metasearcher
    hits: List[Hit] = asyncio.run(MetaSearcher().search(queries))
    # 4. resolve hits to get pdf urls
    resolver_manager = ResolverManager()
    resolved_hits: List[Resolved] = asyncio.run(resolver_manager.resolve_many(hits))

    # deduplicate resolved hits based on pdf url
    seen_urls = set()
    unique_resolved_hits = []
    for resolved in resolved_hits:
        if resolved.pdf_url not in seen_urls:
            seen_urls.add(resolved.pdf_url)
            unique_resolved_hits.append(resolved)

    logger.info(f"Deduplicated {len(resolved_hits)} resolved hits to {len(unique_resolved_hits)} unique PDFs")

    # run the pipeline with deduplicated hits
    asyncio.run(run_pipeline(unique_resolved_hits))

    # testing:
    # for testing
    def dump_hits_to_json(hits: List[Hit]) -> List[dict]:
        # convert hits to json-serializable format
        hits_json = [asdict(hit) for hit in hits]

        # export to json file
        with open('search_results.json', 'w', encoding='utf-8') as f:
            json.dump(hits_json, f, indent=2, ensure_ascii=False)

        print(f"Results exported to search_results.json")

    dump_hits_to_json(hits)

    # for testing
    def dump_resolved_to_json(resolved: List[Resolved]) -> List[dict]:
        # convert resolved to json-serializable format
        resolved_json = [r.to_json() for r in resolved]

        # export to json file
        with open('resolved_results.json', 'w', encoding='utf-8') as f:
            json.dump(resolved_json, f, indent=2, ensure_ascii=False)

        print(f"Resolved results exported to resolved_results.json")

    dump_resolved_to_json(resolved_hits)
