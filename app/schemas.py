# Pydantic şemaları

from pydantic import BaseModel
from typing import Literal
from uuid import UUID
from datetime import datetime

from app.ai.dispatcher import VLMResult, PublicFigure

class PhotoAnalysisResponse(BaseModel):
    photo_id: UUID
    status: str
    description: str | None = None
    environment_type: Literal["indoor", "outdoor", "mixed"] | None = None
    people_count: int | None = None
    possible_event: str | None = None
    primary_object: str | None = None
    secondary_objects: list[str] | None = None
    environment: list[str] | None = None
    attributes: list[str] | None = None
    action: str | None = None
    mood: str | None = None
    use_case: str | None = None
    context: list[str] | None = None
    style: list[str] | None = None
    audience: list[str] | None = None
    public_figures: list[PublicFigure] | None = None
    all_tags: list[str] | None = None

    class Config:
        from_attributes = True
