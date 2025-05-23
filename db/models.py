from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, BigInteger, Enum, ARRAY, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid
from database import Base
from pgvector.sqlalchemy import Vector

class Work(Base):
    __tablename__ = "works"

    work_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doi = Column(String, unique=True)
    title = Column(String)
    year = Column(Integer)
    authors = Column(ARRAY(String))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    files = relationship("File", back_populates="work", cascade="all, delete-orphan")

class File(Base):
    __tablename__ = "files"

    file_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    work_id = Column(UUID(as_uuid=True), ForeignKey("works.work_id", ondelete="CASCADE"), nullable=False)
    uri = Column(String, nullable=False)
    sha256 = Column(String(64), nullable=False, unique=True)
    mime_type = Column(String, nullable=False)
    size_bytes = Column(BigInteger, nullable=False)
    num_pages = Column(Integer)
    licence = Column(String)
    ingested_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    work = relationship("Work", back_populates="files")
    chunks = relationship("Chunk", back_populates="file", cascade="all, delete-orphan")
    summary = relationship("Summary", back_populates="file", uselist=False, cascade="all, delete-orphan")
    ingest_queue = relationship("IngestQueue", back_populates="file", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"

    chunk_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id = Column(UUID(as_uuid=True), ForeignKey("files.file_id", ondelete="CASCADE"), nullable=False)
    page_from = Column(Integer, nullable=False)
    page_to = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(768))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    file = relationship("File", back_populates="chunks")

class Summary(Base):
    __tablename__ = "summaries"

    summary_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id = Column(UUID(as_uuid=True), ForeignKey("files.file_id", ondelete="CASCADE"), nullable=False, unique=True)
    kind = Column(Enum('tl_dr', 'abstract', 'section', name='summary_kind'), nullable=False)
    summary_text = Column(Text, nullable=False)
    generated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    file = relationship("File", back_populates="summary")

class IngestQueue(Base):
    __tablename__ = "ingest_queue"

    ingest_id = Column(BigInteger, primary_key=True)
    file_id = Column(UUID(as_uuid=True), ForeignKey("files.file_id", ondelete="CASCADE"), nullable=False)
    status = Column(String, nullable=False, default='pending')
    retries = Column(Integer, nullable=False, default=0)
    last_error = Column(Text)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    file = relationship("File", back_populates="ingest_queue")