# Pydantic şemaları

from pydantic import BaseModel
from typing import Literal
from uuid import UUID
from datetime import datetime

class VLMResult(BaseModel):
    caption: str
    environment: Literal["indoor", "outdoor", "mixed"]
    activity: str
    people_count: int
    possible_event: str
    summary: str

class PhotoAnalysisResponse(BaseModel):
    photo_id: UUID
    status: str
    caption: str | None = None
    environment: str | None = None
    activity: str | None = None
    people_count: int | None = None
    possible_event: str | None = None
    summary: str | None = None

    class Config:
        from_attributes = True