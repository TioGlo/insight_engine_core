from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector  # For PGVector
from insight_engine_core.config import MODEL_EMBEDDING_DIM

# Import THE Base instance from database/__init__.py
from . import Base  # This should get Base from database/__init__.py

print(f"models.py: Using embedding dimension for Vector type: {MODEL_EMBEDDING_DIM}")


class DataSource(Base):
    """Represents a source of data, e.g., a specific subreddit, a YouTube channel, a website."""
    __tablename__ = "data_sources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False,
                  comment="Unique name for the source, e.g., 'reddit_r_startups', 'youtube_channel_xyz'")
    source_type = Column(String, index=True, nullable=False,
                         comment="e.g., 'reddit', 'youtube', 'web_page', 'google_trends'")
    config = Column(JSON, nullable=True, comment="Source-specific configuration, e.g., API endpoint, subreddit name")
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    raw_data_items = relationship("RawDataItem", back_populates="data_source")

    def __repr__(self):
        return f"<DataSource(id={self.id}, name='{self.name}', type='{self.source_type}')>"


class RawDataItem(Base):
    """Stores raw data fetched from a source before any significant processing or chunking."""
    __tablename__ = "raw_data_items"

    id = Column(Integer, primary_key=True, index=True)
    data_source_id = Column(Integer, ForeignKey("data_sources.id"), nullable=False)

    # Unique identifier within the source, e.g., Reddit post ID, YouTube video ID, URL
    source_internal_id = Column(String, index=True, nullable=False)

    raw_content = Column(JSON, nullable=False, comment="The raw data as fetched, often JSON from APIs")
    retrieved_at = Column(DateTime(timezone=True), server_default=func.now())
    metadata_ = Column("metadata", JSON, nullable=True,
                       comment="Additional metadata from the source, e.g., author, view_count, tags")

    # Ensure a piece of data from a source is stored only once
    __table_args__ = (UniqueConstraint('data_source_id', 'source_internal_id', name='uq_raw_data_source_plus_id'),)

    data_source = relationship("DataSource", back_populates="raw_data_items")
    processed_texts = relationship("ProcessedText", back_populates="raw_data_item")

    def __repr__(self):
        return f"<RawDataItem(id={self.id}, data_source_id={self.data_source_id}, source_internal_id='{self.source_internal_id}')>"


class ProcessedText(Base):
    """Represents a piece of text extracted and cleaned from a RawDataItem, ready for chunking or direct embedding."""
    __tablename__ = "processed_texts"

    id = Column(Integer, primary_key=True, index=True)
    raw_data_item_id = Column(Integer, ForeignKey("raw_data_items.id"), nullable=False)

    # Type of text, e.g., 'title', 'body', 'comment', 'transcript'
    text_type = Column(String, index=True, nullable=True, default="body")
    cleaned_text = Column(Text, nullable=False)

    # Optional: if this processed text itself is directly embeddable without further chunking
    embedding = Column(Vector(MODEL_EMBEDDING_DIM), nullable=True)

    processing_log = Column(JSON, nullable=True, comment="Log of processing steps applied")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    raw_data_item = relationship("RawDataItem", back_populates="processed_texts")
    text_chunks = relationship("TextChunk", back_populates="processed_text_source")

    # Index for the embedding if present
    __table_args__ = (
        Index(
            'idx_processed_text_embedding',
            embedding,  # This will only work if the column is not nullable or handled carefully
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_cosine_ops'},
            postgresql_where=(embedding.isnot(None))  # Index only non-null embeddings
        ),
    )

    def __repr__(self):
        return f"<ProcessedText(id={self.id}, raw_data_item_id={self.raw_data_item_id}, type='{self.text_type}', len='{len(self.cleaned_text)}')>"


class TextChunk(Base):
    """Stores individual text chunks derived from ProcessedText, along with their embeddings."""
    __tablename__ = "text_chunks"

    id = Column(Integer, primary_key=True, index=True)
    # Link to the ProcessedText from which this chunk originated
    processed_text_source_id = Column(Integer, ForeignKey("processed_texts.id"), nullable=False)

    chunk_text = Column(Text, nullable=False)
    # The actual vector embedding for this chunk
    embedding = Column(Vector(MODEL_EMBEDDING_DIM), nullable=False)

    chunk_order = Column(Integer, nullable=False, default=0,
                         comment="Order of this chunk within its parent ProcessedText")
    metadata_ = Column("metadata", JSON, nullable=True, comment="Chunk-specific metadata, e.g., page number, section")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    processed_text_source = relationship("ProcessedText", back_populates="text_chunks")

    __table_args__ = (
        Index(
            'idx_text_chunk_embedding_hnsw',  # Unique index name
            embedding,
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
    )

    def __repr__(self):
        return f"<TextChunk(id={self.id}, processed_text_id={self.processed_text_source_id}, order={self.chunk_order}, len='{len(self.chunk_text)}')>"


# --- Application-Specific Models (Example for Niche Hunter - these might live in niche_hunter_app later) ---
# For now, let's keep them here to ensure the core RAG can support them.

class IdentifiedNiche(Base):
    """Represents a potential niche identified by the system."""
    __tablename__ = "identified_niches"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    seed_keywords = Column(JSON, nullable=True, comment="Keywords that led to this niche")
    # Link to relevant DataSources or RawDataItems that informed this niche
    # This could be a many-to-many relationship or a JSONB array of IDs

    # Scores and analysis results
    trend_score = Column(Integer, nullable=True)
    competition_score = Column(Integer, nullable=True)  # Lower might be better
    monetization_score = Column(Integer, nullable=True)
    overall_opportunity_score = Column(Integer, nullable=True, index=True)

    analysis_summary = Column(Text, nullable=True)  # LLM generated summary
    status = Column(String, default="new", index=True)  # e.g., new, analyzing, validated, rejected

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_analyzed_at = Column(DateTime(timezone=True), onupdate=func.now())

    # content_outlines = relationship("ContentOutline", back_populates="niche")

    def __repr__(self):
        return f"<IdentifiedNiche(id={self.id}, name='{self.name}', score='{self.overall_opportunity_score}')>"

# We can add ContentOutline, NicheMetrics etc. later as needed.
