r"""
Otomatik yedekleme - PhotoAI'nin gercek bir "backup sistemi" olmasi icin
(bkz. sohbet: elle alinmis tek seferlik bir yedek, gercek tanimi
karsilamiyordu - "ayri konum" + "duzenli/otomatik" + "test edilmis geri
yukleme" gerekiyor).

Bu script uc seyi yapar:
  1. PostgreSQL'in tam dokumunu alir (pg_dump, custom format).
  2. uploads/ klasorunu (fotograflar + yuz kirpimlari) sikistirip ekler.
  3. BACKUP_ROOT altinda, eski yedekleri (varsayilan: son 7'den fazlasini)
     siler - sinirsiz buyumeyi engeller.

BACKUP_ROOT: varsayilan olarak Masaustu'nde gecici bir klasor - Google
Drive masaustu uygulamasi kurulup senkronize bir klasor hazir oldugunda
--backup-root ile (ya da PHOTOAI_BACKUP_ROOT ortam degiskeniyle) o klasore
yonlendirilmeli. O ana kadar yedekler "ayri konum" sartini TAM karsilamiyor
(ayni diskte) - bu bilinen, gecici bir durum.

Zamanlanmis calistirma (Windows Task Scheduler):
    schtasks /create /tn "PhotoAI Daily Backup" ^
        /tr "\"C:\Users\erdem\Desktop\photoai-backend\venv\Scripts\python.exe\" \"C:\Users\erdem\Desktop\photoai-backend\scripts\backup_system.py\"" ^
        /sc daily /st 03:00 /f

Kullanim:
    python scripts/backup_system.py
    python scripts/backup_system.py --backup-root "C:\Users\erdem\Google Drive\PhotoAI-Backups"
    python scripts/backup_system.py --keep 14
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BACKUP_ROOT = Path(r"C:\Users\erdem\Desktop\PhotoAI-Backups")  # GECICI - Drive kurulunca degistirilecek
PG_DUMP_EXE = r"C:\Program Files\PostgreSQL\17\bin\pg_dump.exe"


def _load_database_url() -> str:
    """app/.env icinden DATABASE_URL'i okur (app'i import etmeden, hafif)."""
    env_path = BACKEND_ROOT / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(".env icinde DATABASE_URL bulunamadi")


def backup_database(dest_dir: Path) -> Path:
    dump_path = dest_dir / "photoai_db.dump"
    db_url = _load_database_url()
    result = subprocess.run(
        [PG_DUMP_EXE, db_url, "-F", "c", "-f", str(dump_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pg_dump basarisiz: {result.stderr}")
    return dump_path


def backup_uploads(dest_dir: Path) -> Path | None:
    uploads_dir = BACKEND_ROOT / "uploads"
    if not uploads_dir.exists():
        return None
    archive_base = dest_dir / "uploads"
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=str(uploads_dir))
    return Path(archive_path)


def cleanup_old_backups(backup_root: Path, keep: int) -> list[Path]:
    """backup_root altindaki tarih-damgali klasorlerden en yeni `keep` tanesini
    birakip gerisini siler."""
    candidates = sorted(
        [d for d in backup_root.iterdir() if d.is_dir() and d.name[:8].isdigit()],
        key=lambda d: d.name,
        reverse=True,
    )
    removed = []
    for old in candidates[keep:]:
        shutil.rmtree(old, ignore_errors=True)
        removed.append(old)
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backup-root",
        default=os.getenv("PHOTOAI_BACKUP_ROOT", str(DEFAULT_BACKUP_ROOT)),
        help="Yedeklerin yazilacagi kok klasor (idealde Google Drive senkronize klasoru)",
    )
    parser.add_argument("--keep", type=int, default=7, help="Kac gunluk yedek saklansin (varsayilan: 7)")
    parser.add_argument("--skip-uploads", action="store_true", help="Sadece veritabanini yedekle, fotograflari atla (hizli test icin)")
    args = parser.parse_args()

    backup_root = Path(args.backup_root)
    backup_root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = backup_root / stamp
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Yedek hedefi: {dest_dir}")

    try:
        dump_path = backup_database(dest_dir)
        dump_size_mb = dump_path.stat().st_size / (1024 * 1024)
        print(f"  Veritabani  : {dump_path.name} ({dump_size_mb:.2f} MB)")
    except Exception as e:
        print(f"  [HATA] Veritabani yedeklenemedi: {e}")
        shutil.rmtree(dest_dir, ignore_errors=True)
        sys.exit(1)

    if not args.skip_uploads:
        try:
            archive_path = backup_uploads(dest_dir)
            if archive_path:
                archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
                print(f"  Fotograflar : {archive_path.name} ({archive_size_mb:.2f} MB)")
            else:
                print("  Fotograflar : uploads/ bulunamadi, atlandi")
        except Exception as e:
            print(f"  [UYARI] Fotograflar yedeklenemedi (veritabani yedegi yine de gecerli): {e}")

    manifest = dest_dir / "MANIFEST.txt"
    manifest.write_text(
        f"PhotoAI otomatik yedek\n"
        f"Alinma zamani: {datetime.now().isoformat()}\n"
        f"Kaynak: {BACKEND_ROOT}\n",
        encoding="utf-8",
    )

    removed = cleanup_old_backups(backup_root, args.keep)
    if removed:
        print(f"  Eski yedekler silindi ({len(removed)} adet, son {args.keep} tutuluyor): "
              + ", ".join(d.name for d in removed))

    print("Tamamlandi.")


if __name__ == "__main__":
    main()
