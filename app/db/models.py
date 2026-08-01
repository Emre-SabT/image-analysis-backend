# tablolar

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Integer, text
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
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
    description = Column(Text)
    environment_type = Column(String)  # indoor | outdoor | mixed
    people_count = Column(Integer)
    possible_event = Column(String)
    primary_object = Column(String)
    secondary_objects = Column(ARRAY(String), server_default=text("'{}'"))
    environment = Column(ARRAY(String), server_default=text("'{}'"))  # mekan/ortam etiketleri (eski environment_type ile karistirma)
    attributes = Column(ARRAY(String), server_default=text("'{}'"))
    action = Column(String)
    mood = Column(String)
    use_case = Column(String)
    context = Column(ARRAY(String), server_default=text("'{}'"))
    style = Column(ARRAY(String), server_default=text("'{}'"))
    audience = Column(ARRAY(String), server_default=text("'{}'"))
    public_figures = Column(JSONB, server_default=text("'[]'::jsonb"))
    all_tags = Column(ARRAY(String), server_default=text("'{}'"))
    model_name = Column(String)
    analyzed_at = Column(DateTime)