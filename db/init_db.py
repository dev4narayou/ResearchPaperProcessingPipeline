from sqlalchemy import text
from .database import engine

def init_db():
    # Create extensions
    with engine.connect() as conn:
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS "vector"'))
        conn.commit()

    # Create tables
    from .models import Base
    Base.metadata.create_all(bind=engine)

    # Create indexes
    with engine.connect() as conn:
        conn.execute(text('''
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        '''))

        conn.execute(text('''
            CREATE INDEX IF NOT EXISTS idx_ingest_queue_pending
            ON ingest_queue (status)
            WHERE status = 'pending'
        '''))
        conn.commit()

if __name__ == "__main__":
    init_db()