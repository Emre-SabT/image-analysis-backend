# Pydantic şemaları

from pydantic import BaseModel, EmailStr, Field
from typing import Literal
from uuid import UUID
from datetime import datetime

from app.ai.dispatcher import VLMResult, PublicFigure


# --- Kimlik dogrulama / kullanici yonetimi ---

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class UserCreateRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    display_name: str
    role: Literal["admin", "editor", "viewer"] = "viewer"


class UserUpdateRequest(BaseModel):
    display_name: str | None = None
    role: Literal["admin", "editor", "viewer"] | None = None
    is_active: bool | None = None


class UserResponse(BaseModel):
    id: UUID
    email: str
    display_name: str
    role: str
    is_active: bool
    created_at: datetime | None = None

    class Config:
        from_attributes = True


class ClusterLabelRequest(BaseModel):
    display_name: str


class IdentityRef(BaseModel):
    kind: Literal["person", "cluster"]
    id: UUID


class RejectMergeRequest(BaseModel):
    """POST /identities/reject-merge - "bunlar ayni kisi degil" geri bildirimi."""
    identities: list[IdentityRef]


class IdentityMergeRequest(BaseModel):
    """POST /identities/merge - iki klasoru (kisi ya da isimsiz kume, herhangi
    bir kombinasyon) birlestirir. Sonuc target_kind/target_id'de kalir."""
    target_kind: Literal["person", "cluster"]
    target_id: UUID
    source_kind: Literal["person", "cluster"]
    source_id: UUID


class FaceReassignRequest(BaseModel):
    person_id: UUID | None = None  # None = kisiden ayir (kimliksiz havuza don)

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
