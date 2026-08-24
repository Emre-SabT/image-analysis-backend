import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.core.settings import settings
from app.db import jobs_repository
from app.db.models import User
from app.db.session import get_db

router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=[Depends(get_current_user)])


@router.get("/queue-status")
def queue_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Kullanicinin kuyrugu - IS TIPI BAZINDA ayri sayim ve ayri tahmini sure.

    face (~0,13 sn) ve vlm (~8-20 sn) sureleri buyuklukce farkli oldugu icin
    KARISIK TEK ORTALAMA verilmiyor. Tahmini sure sabit bir varsayim degil,
    son N tamamlanmis isin OLCULEN ortalamasindan hesaplaniyor.
    """
    by_type = jobs_repository.queue_status_by_type(
        current_user.id, settings.JOB_ETA_SAMPLE_SIZE, session=db
    )
    return {
        "by_type": by_type,
        "total_queued": sum(v["queued"] for v in by_type.values()),
    }


@router.get("/{job_id}")
def get_job(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    status = jobs_repository.get_status(job_id, session=db)
    if status is None:
        raise HTTPException(status_code=404, detail="Is bulunamadi")
    return status
