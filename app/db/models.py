# tablolar

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Integer
from sqlalchemy.dialects.postgresql import UUID
from app.db.session import Base

class Photo(Base):
    __tablename__ = "photos"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    status = Column(String, default="uploaded")  # uploaded -> processing -> analyzed -> failed
    created_at = Column(DateTime, default=datetime.utcnow)

class PhotoAnalysis(Base):
    __tablename__ = "photo_analysis"
    photo_id = Column(UUID(as_uuid=True), ForeignKey("photos.id"), primary_key=True)
    caption = Column(Text)
    environment = Column(String)
    activity = Column(String)
    people_count = Column(Integer)
    possible_event = Column(String)
    summary = Column(Text)
    model_name = Column(String)
    analyzed_at = Column(DateTime)