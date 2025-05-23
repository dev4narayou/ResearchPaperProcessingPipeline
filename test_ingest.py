import asyncio
import aiohttp
import hashlib
import logging
from pathlib import Path
from typing import List

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# create queues for the pipeline
download_queue = asyncio.Queue()
process_queue = asyncio.Queue()

# create storage directory
STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)

async def download_worker():
    """worker that downloads pdfs from urls"""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        while True:
            try:
                resolved = await download_queue.get()
                try:
                    logger.info(f"Downloading {resolved.pdf_url}")
                    async with session.get(resolved.pdf_url) as response:
                        if response.status == 200:
                            content = await response.read()
                            # Calculate SHA256 hash
                            sha256 = hashlib.sha256(content).hexdigest()
                            # Save to storage
                            file_path = STORAGE_DIR / f"{sha256}.pdf"
                            file_path.write_bytes(content)

                            # Queue for processing
                            await process_queue.put({
                                'resolved': resolved,
                                'file_path': str(file_path),
                                'sha256': sha256,
                                'size': len(content)
                            })
                            logger.info(f"Downloaded and queued {resolved.pdf_url}")
                        else:
                            logger.error(f"Failed to download {resolved.pdf_url}: {response.status}")
                except Exception as e:
                    logger.error(f"Error downloading {resolved.pdf_url}: {str(e)}")
                finally:
                    download_queue.task_done()
            except asyncio.CancelledError:
                break

async def process_worker():
    """worker that processes downloaded pdfs"""
    while True:
        try:
            item = await process_queue.get()
            try:
                resolved = item['resolved']
                file_path = item['file_path']
                sha256 = item['sha256']
                size = item['size']

                logger.info(f"Processing {file_path}")
                # todo: add your pdf processing logic here
                # for example:
                # - extract text
                # - generate embeddings
                # - store in database

                logger.info(f"Processed {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
            finally:
                process_queue.task_done()
        except asyncio.CancelledError:
            break

async def run_pipeline(resolved_hits: List[Resolved], num_download_workers: int = 3, num_process_workers: int = 2):
    """run the download and process pipeline"""
    # start workers
    download_workers = [asyncio.create_task(download_worker())
                       for _ in range(num_download_workers)]
    process_workers = [asyncio.create_task(process_worker())
                      for _ in range(num_process_workers)]

    try:
        # queue all resolved hits for download
        for resolved in resolved_hits:
            await download_queue.put(resolved)

        # wait for all downloads to complete
        await download_queue.join()
        # wait for all processing to complete
        await process_queue.join()
    finally:
        # cancel workers
        for worker in download_workers + process_workers:
            worker.cancel()
        # wait for workers to finish
        await asyncio.gather(*download_workers, *process_workers, return_exceptions=True)