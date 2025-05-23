import asyncio
import asyncpg
from datetime import datetime
import numpy as np
from config.config import settings
import json

async def seed_test_data():
    # Connect to the database
    conn = await asyncpg.connect(settings.DATABASE_URL)

    try:
        # Insert test papers
        papers = [
            {
                'doi': '10.1234/test1',
                'title': 'The Effects of Creatine on Exercise Performance',
                'authors': ['John Smith', 'Jane Doe'],
                'year': 2023,
                'pdf_url': 'https://example.com/papers/test1.pdf',
                'sha256': 'test1_hash',
                'source': 'test'
            },
            {
                'doi': '10.1234/test2',
                'title': 'Creatine Supplementation in Athletes',
                'authors': ['Alice Johnson', 'Bob Wilson'],
                'year': 2022,
                'pdf_url': 'https://example.com/papers/test2.pdf',
                'sha256': 'test2_hash',
                'source': 'test'
            }
        ]

        for paper in papers:
            paper_id = await conn.fetchval("""
                INSERT INTO papers (doi, title, authors, year, pdf_url, sha256, source)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (doi) DO UPDATE SET
                    title=EXCLUDED.title,
                    authors=EXCLUDED.authors,
                    year=EXCLUDED.year,
                    pdf_url=EXCLUDED.pdf_url,
                    sha256=EXCLUDED.sha256,
                    source=EXCLUDED.source
                RETURNING id
            """,
                paper['doi'], paper['title'], paper['authors'],
                paper['year'], paper['pdf_url'], paper['sha256'],
                paper['source']
            )

            # Insert summaries
            summaries = [
                ('executive', 'This study examines the effects of creatine supplementation on athletic performance.'),
                ('key_findings', 'Creatine supplementation showed significant improvements in strength and power output.'),
                ('methodology', 'Double-blind, placebo-controlled study with 50 participants.')
            ]

            for summary_type, content in summaries:
                await conn.execute("""
                    INSERT INTO summaries (paper_id, summary_type, content)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (paper_id, summary_type) DO UPDATE SET content=EXCLUDED.content
                """, paper_id, summary_type, content)

            # Insert embeddings (random vectors for testing)
            embedding_types = ['title', 'abstract', 'full_text']
            for embedding_type in embedding_types:
                embedding = np.random.rand(1536).tolist()  # OpenAI's embedding dimension
                embedding_str = str(embedding)  # Convert list to string for pgvector
                await conn.execute("""
                    INSERT INTO embeddings (paper_id, embedding_type, embedding)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (paper_id, embedding_type) DO NOTHING
                """, paper_id, embedding_type, embedding_str)

            # Insert metadata
            metadata = [
                ('keywords', ['creatine', 'exercise', 'performance']),
                ('abstract', 'A comprehensive study on creatine supplementation...'),
                ('citations', 42)
            ]
            for key, value in metadata:
                value_json = json.dumps(value)
                await conn.execute("""
                    INSERT INTO paper_metadata (paper_id, key, value)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (paper_id, key) DO UPDATE SET value=EXCLUDED.value
                """, paper_id, key, value_json)

        # Insert relationships between papers
        paper_ids = await conn.fetch("SELECT id FROM papers ORDER BY id")
        if len(paper_ids) >= 2:
            source_id = paper_ids[0]['id']
            target_id = paper_ids[1]['id']
            await conn.execute("""
                INSERT INTO paper_relationships
                (source_paper_id, target_paper_id, relationship_type, confidence)
                VALUES ($1, $2, $3, $4)
            """, source_id, target_id, 'cites', 0.95)

        print("Test data seeded successfully!")

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(seed_test_data())