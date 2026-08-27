"""Sistem-geneli, oturum acmis HERKESE (viewer dahil) acik salt-okunur
uclar - `/health` gibi rol kisiti gerektirmeyen ama Bearer token
isteyen bilgiler (BACKEND_IHTIYACLARI.md #9)."""

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.models import Photo
from app.db.session import get_db
from app.services import activity_log_service, photo_service

router = APIRouter(prefix="/system", tags=["system"], dependencies=[Depends(get_current_user)])


@router.get("/storage")
def storage_usage(db: Session = Depends(get_db)):
    """GET /system/storage (BACKEND_IHTIYACLARI.md #9).

    `photos_total_bytes` - fotograflarin GERCEK toplam boyutu: her
    `Photo.storage_path`'in diskteki GERCEK dosya boyutu (`Path.stat()`)
    tek tek toplanir. BILINCLI OLARAK ayri bir `file_size_bytes` sutunu/
    migration GEREKMEDI - 452 fotografta bu birkaç ms'lik, önbelleklenmesi
    gerekmeyen ucuz bir IO taramasi; ayrica bir sutuna GUVENMEK yerine
    diskten HER SEFERINDE gercek deger okunur, kayit/dosya arasinda
    sapma (ör. dosya elle silinmis/degismis) OLASI DEGILDIR. Silinmis bir
    fotografin dosyasi zaten `photo_service.delete_photo` ile birlikte
    silindigi icin burada ekstra bir "eski kayit" filtrelemesi gerekmez;
    yine de eksik/erisilemeyen bir dosya varsa `OSError` sessizce atlanir
    (tek bir bozuk kayit yuzunden tum endpoint patlamamali).

    `disk_*` alanlari - `uploads/` klasorunun bulundugu DISKIN TAMAMININ
    kullanimi (`shutil.disk_usage`, ayri bir tablo/arka plan isi GEREKMEZ).
    Bu, `photos_total_bytes`'tan FARKLI bir olcum: diskteki her sey (OS,
    veritabani, diger uygulamalar) dahildir - frontend'in birincil olarak
    gosterdigi sayi `photos_total_bytes` olmali (yaniltici olmasin diye,
    bkz. kullanici geri bildirimi), `disk_total_bytes` yalnizca "fotograf
    verisi diskin ne kadarini kapliyor" oranini hesaplamak icin BAGLAM
    olarak sunulur.
    """
    photos_total_bytes = 0
    for (storage_path,) in db.query(Photo.storage_path).all():
        try:
            photos_total_bytes += Path(storage_path).stat().st_size
        except OSError:
            pass

    disk = shutil.disk_usage(photo_service.UPLOAD_DIR)
    return {
        "photos_total_bytes": photos_total_bytes,
        "disk_used_bytes": disk.used,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }


@router.get("/activity")
def activity_feed(limit: int = 50, photo_id: uuid.UUID | None = None, db: Session = Depends(get_db)):
    """GET /system/activity - kurumsal ORTAK havuzdaki TUM kullanicilarin
    islem gunlugu (silme, birlestirme, albume ekleme/cikarma, yeniden
    adlandirma, yuz atama, vb.) - kullanici bazli filtre YOK (bkz.
    app/services/activity_log_service.py docstring'i).

    `photo_id` verilirse - Fotograf Detayi'nin "Islem Gecmisi" sekmesi icin -
    yalnizca o fotografi ilgilendiren olaylar donulur (bkz.
    activity_log_service.list_activity docstring'i, kapsam siniri dahil)."""
    return activity_log_service.list_activity(db, limit=limit, photo_id=photo_id)
