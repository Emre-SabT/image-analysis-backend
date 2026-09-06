"""Inbox mimarisinden ONCE yuklenmis fotograflari uploads/ -> uploads/stored/ tasir.

NEDEN GEREKLI DEGIL, NEDEN YINE DE ISTENEBILIR
----------------------------------------------
Sistem bu goc OLMADAN da dogru calisir: her tuketici yolu dosyayi
`Photo.storage_path` uzerinden DB'den okur (face_service, dispatcher,
system router, silme), hicbir yerde sabit kodlu dizin yoktur. Eski kayitlar
"uploads/{uuid}.ext", yeniler "uploads/stored/{uuid}.ext" gosterir ve ikisi
de calisir.

Gocun tek kazanimi DUZEN: yedekleme/arsivleme kurallari, disk kotasi ve
"hangi klasor ne ise yarar" sorusu tek bir yerde toplanir. Bu yuzden goc
ISTEGE BAGLIDIR ve VARSAYILAN OLARAK CALISMAZ (--apply gerekir).

CRASH GUVENLIGI - neden tasima degil KOPYALA/DOGRULA/GUNCELLE/SIL
-----------------------------------------------------------------
Naif bir "once tasi sonra UPDATE" (ya da tersi) her iki sirada da yarim
kalabilecek bir aralik birakir: DB bir yeri, dosya baska yeri gosterir ve
fotograf ERISILEMEZ hale gelir. Bu betik her adimda guvenli olan sirayi
kullanir:

    1. stored/'a KOPYALA (orijinal yerinde durur)
    2. SHA-256 ile DOGRULA (kopya bozuksa hicbir sey degismez)
    3. storage_path'i GUNCELLE + commit
    4. orijinali SIL

Hangi adimda kesilirse kesilsin fotograf her an EN AZ BIR gecerli yoldan
erisilebilir; betik tekrar calistirilinca kaldigi yerden devam eder
(idempotent).

KULLANIM
--------
    python scripts/migrate_photos_to_stored.py            # kuru kosum (varsayilan)
    python scripts/migrate_photos_to_stored.py --apply    # gercekten uygula
    python scripts/migrate_photos_to_stored.py --apply --limit 50
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402
from app.services.photo_service import STORED_DIR, UPLOAD_DIR  # noqa: E402

_CHUNK = 1024 * 1024


def _sha256(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            d.update(chunk)
    return d.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Gercekten uygula (verilmezse yalnizca rapor)")
    ap.add_argument("--limit", type=int, default=0, help="En fazla N fotograf")
    args = ap.parse_args()

    STORED_DIR.mkdir(parents=True, exist_ok=True)
    db = SessionLocal()
    stats = {"zaten_stored": 0, "tasinacak": 0, "dosya_yok": 0,
             "tasindi": 0, "hata": 0, "atlandi_hedef_var": 0}
    try:
        rows = db.execute(text(
            "SELECT id, filename, storage_path FROM photos ORDER BY created_at"
        )).fetchall()
        print(f"Toplam fotograf: {len(rows)}")
        print(f"Hedef dizin    : {STORED_DIR.resolve()}")
        print(f"Mod            : {'UYGULA' if args.apply else 'KURU KOSUM (degisiklik yok)'}")
        print("-" * 72)

        processed = 0
        for pid, filename, storage_path in rows:
            src = Path(storage_path)
            try:
                if src.parent.resolve() == STORED_DIR.resolve():
                    stats["zaten_stored"] += 1
                    continue
                if not src.exists():
                    stats["dosya_yok"] += 1
                    print(f"  ! dosya yok  : {filename}  ({storage_path})")
                    continue

                dst = STORED_DIR / src.name
                if dst.exists() and _sha256(dst) == _sha256(src):
                    # Onceki yarim kosumdan kalmis gecerli kopya: DB'yi
                    # guncelleyip orijinali silmek yeterli.
                    stats["atlandi_hedef_var"] += 1
                else:
                    stats["tasinacak"] += 1

                processed += 1
                if args.limit and processed > args.limit:
                    break
                if not args.apply:
                    continue

                # 1) KOPYALA (orijinal yerinde kalir)
                if not dst.exists():
                    shutil.copy2(src, dst)
                # 2) DOGRULA
                if _sha256(dst) != _sha256(src):
                    dst.unlink(missing_ok=True)
                    raise RuntimeError("kopya dogrulanamadi (hash uyusmadi)")
                # 3) DB'yi GUNCELLE
                db.execute(
                    text("UPDATE photos SET storage_path = :sp WHERE id = :id"),
                    {"sp": str(dst), "id": str(pid)},
                )
                db.commit()
                # 4) ORIJINALI SIL
                src.unlink(missing_ok=True)
                stats["tasindi"] += 1
            except Exception as e:  # tek fotograf patlarsa digerleri devam etsin
                db.rollback()
                stats["hata"] += 1
                print(f"  ! HATA       : {filename}: {type(e).__name__}: {e}")

        print("-" * 72)
        for k, v in stats.items():
            print(f"  {k:20s}: {v}")
        if not args.apply:
            print("\nKURU KOSUM - hicbir dosya/kayit degistirilmedi.")
            print("Uygulamak icin: python scripts/migrate_photos_to_stored.py --apply")
    finally:
        db.close()
    return 1 if stats["hata"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
