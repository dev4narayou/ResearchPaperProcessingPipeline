-- Enable pgvector extension for vector embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Papers table to store basic paper information
CREATE TABLE papers (
    id SERIAL PRIMARY KEY,
    doi TEXT UNIQUE,
    title TEXT NOT NULL,
    authors TEXT[],
    year INTEGER,
    pdf_url TEXT,
    file_path TEXT,
    sha256 TEXT UNIQUE,
    source TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index on DOI for fast lookups
CREATE INDEX idx_papers_doi ON papers(doi);

-- Summaries table to store different types of summaries
CREATE TABLE summaries (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    summary_type TEXT NOT NULL, -- e.g., 'executive', 'key_findings', 'methodology'
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(paper_id, summary_type)
);

-- Embeddings table to store vector embeddings for semantic search
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    embedding_type TEXT NOT NULL, -- e.g., 'title', 'abstract', 'full_text'
    embedding vector(1536), -- OpenAI's embedding dimension
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(paper_id, embedding_type)
);

-- Create index for vector similarity search
CREATE INDEX idx_embeddings_vector ON embeddings USING ivfflat (embedding vector_cosine_ops);

-- Paper relationships table for knowledge graph
CREATE TABLE paper_relationships (
    id SERIAL PRIMARY KEY,
    source_paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    target_paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL, -- e.g., 'cites', 'cited_by', 'related_to'
    confidence FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_paper_id, target_paper_id, relationship_type)
);

-- Create indexes for relationship lookups
CREATE INDEX idx_relationships_source ON paper_relationships(source_paper_id);
CREATE INDEX idx_relationships_target ON paper_relationships(target_paper_id);

-- Metadata table for additional paper information
CREATE TABLE paper_metadata (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(paper_id, key)
);

-- Create index for metadata lookups
CREATE INDEX idx_metadata_paper_key ON paper_metadata(paper_id, key);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_papers_updated_at
    BEFORE UPDATE ON papers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_summaries_updated_at
    BEFORE UPDATE ON summaries
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_metadata_updated_at
    BEFORE UPDATE ON paper_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to find similar papers using vector similarity
CREATE OR REPLACE FUNCTION find_similar_papers(
    query_embedding vector(1536),
    similarity_threshold float DEFAULT 0.7,
    limit_count int DEFAULT 10
)
RETURNS TABLE (
    paper_id integer,
    title text,
    similarity float
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id,
        p.title,
        1 - (e.embedding <=> query_embedding) as similarity
    FROM embeddings e
    JOIN papers p ON p.id = e.paper_id
    WHERE 1 - (e.embedding <=> query_embedding) > similarity_threshold
    ORDER BY similarity DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;
