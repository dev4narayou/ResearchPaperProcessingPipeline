from sqlalchemy import create_engine, select, insert
from sqlalchemy.orm import Session, registry
from uuid import uuid4
from core.Hit import Hit
import os
from typing import List, Optional, TypeVar, Generic, Type
from .models import Work, File, Chunk, Summary, IngestQueue, SearchHit


engine = create_engine(os.getenv("DATABASE_URL"), future=True)

def seen_before(hit: Hit) -> bool:
    with Session(engine) as s:
        if hit.doi:
            return s.scalar(select(Work).where(Work.doi == hit.doi)) is not None
        return s.scalar(select(SearchHit).where(SearchHit.url == hit.url)) is not None

def save_hit(hit: Hit, topic: str):
    with Session(engine) as s:
        work_id = uuid4()
        if hit.doi:
            s.execute(insert(Work).values(work_id=work_id, doi=hit.doi))
        s.execute(insert(SearchHit).values(hit_id=uuid4(),
                                           work_id=work_id,
                                           url=hit.url,
                                           provider=hit.provider,
                                           topic=topic))
        s.commit()

T = TypeVar('T')

class BaseRepository(Generic[T]):
    def __init__(self, model: Type[T]):
        self.model = model

    def get(self, db: Session, id: uuid.UUID) -> Optional[T]:
        return db.query(self.model).filter(self.model.work_id == id).first()

    def get_all(self, db: Session, skip: int = 0, limit: int = 100) -> List[T]:
        return db.query(self.model).offset(skip).limit(limit).all()

    def create(self, db: Session, obj_in: T) -> T:
        db.add(obj_in)
        db.commit()
        db.refresh(obj_in)
        return obj_in

    def update(self, db: Session, db_obj: T, obj_in: dict) -> T:
        for field, value in obj_in.items():
            setattr(db_obj, field, value)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def delete(self, db: Session, id: uuid.UUID) -> bool:
        obj = db.query(self.model).filter(self.model.work_id == id).first()
        if obj:
            db.delete(obj)
            db.commit()
            return True
        return False

class WorkRepository(BaseRepository[Work]):
    def get_by_doi(self, db: Session, doi: str) -> Optional[Work]:
        return db.query(Work).filter(Work.doi == doi).first()

class FileRepository(BaseRepository[File]):
    def get_by_sha256(self, db: Session, sha256: str) -> Optional[File]:
        return db.query(File).filter(File.sha256 == sha256).first()

    def get_by_work_id(self, db: Session, work_id: uuid.UUID) -> List[File]:
        return db.query(File).filter(File.work_id == work_id).all()

class ChunkRepository(BaseRepository[Chunk]):
    def get_by_file_id(self, db: Session, file_id: uuid.UUID) -> List[Chunk]:
        return db.query(Chunk).filter(Chunk.file_id == file_id).all()

    def get_similar_chunks(self, db: Session, embedding: List[float], limit: int = 10) -> List[Chunk]:
        # Note: This requires pgvector extension and proper indexing
        return db.query(Chunk).order_by(Chunk.embedding.cosine_distance(embedding)).limit(limit).all()

class SummaryRepository(BaseRepository[Summary]):
    def get_by_file_id(self, db: Session, file_id: uuid.UUID) -> Optional[Summary]:
        return db.query(Summary).filter(Summary.file_id == file_id).first()

class IngestQueueRepository(BaseRepository[IngestQueue]):
    def get_pending(self, db: Session, limit: int = 100) -> List[IngestQueue]:
        return db.query(IngestQueue).filter(IngestQueue.status == 'pending').limit(limit).all()

    def update_status(self, db: Session, ingest_id: int, status: str, error: Optional[str] = None) -> Optional[IngestQueue]:
        queue_item = db.query(IngestQueue).filter(IngestQueue.ingest_id == ingest_id).first()
        if queue_item:
            queue_item.status = status
            if error:
                queue_item.last_error = error
                queue_item.retries += 1
            db.commit()
            db.refresh(queue_item)
        return queue_item
