"""photos.content_hash uzerinde UNIQUE index

Revision ID: b4d7e2a9c153
Revises: a2c4e6f8b0d1
Create Date: 2026-09-04

NEDEN: fotograf tekillestirmesi bugune kadar YALNIZCA uygulama katmaninda
("once SELECT sonra INSERT") yapiliyordu. Bu, yuklemeler sirali oldugu
surece yeterliydi - ki modeldeki eski yorum da tam olarak bu varsayima
dayaniyordu. Iki sey degisti:

  1. Frontend yukleme eszamanliligi sabit 3'ten adaptif 2-12'ye cikti.
  2. Photo satirini olusturma isi, HTTP istegi icinden ayri bir ingestion
     donguse tasindi (inbox -> ingestion mimarisi).

Ikisi birlikte, ayni icerik icin iki es zamanli yolun da "kayit yok" gorup
INSERT etmesini MUMKUN kildi. Uygulama tarafinda icerik-scoped advisory
kilit eklendi (app/db/locks.py: acquire_content_lock), ama tekillik gibi bir
veri butunlugu kurali icin SON SOZ veritabaninin olmali.

GUVENLIK KONTROLU (migration yazilmadan once uretim verisinde olculdu):
    SELECT count(*), count(DISTINCT content_hash), count(*) FILTER (WHERE content_hash IS NULL)
    FROM photos;
    -> 483 satir / 483 tekil hash / 0 NULL  => 0 yinelenen grup.

Modeldeki eski yorumun bahsettigi "7 grup yinelenen fotograf" o zamandan
beri temizlenmis. upgrade() yine de kisiti eklemeden ONCE kontrol eder ve
yinelenen varsa ANLASILIR bir hatayla durur - sessizce patlayan bir DDL
yerine, operatore ne yapacagini soyleyen bir mesaj.

NULL davranisi: PostgreSQL'de unique index birden fazla NULL'a izin verir,
dolayisiyla hash'i henuz doldurulmamis eski kayitlar etkilenmez.
"""

import sqlalchemy as sa
from alembic import op

revision = "b4d7e2a9c153"
down_revision = "a2c4e6f8b0d1"
branch_labels = None
depends_on = None

# Duz (unique olmayan) index, models.py'de index=True oldugu icin
# a7f3... goc zincirinde bu adla olusmustu; unique olani ekleyip
# gereksiz olani dusuruyoruz.
_OLD_INDEX = "ix_photos_content_hash"
_NEW_INDEX = "uq_photos_content_hash"


def upgrade() -> None:
    conn = op.get_bind()

    dupes = conn.execute(
        sa.text(
            """
            SELECT content_hash, count(*) AS n
            FROM photos
            WHERE content_hash IS NOT NULL
            GROUP BY content_hash
            HAVING count(*) > 1
            ORDER BY n DESC
            LIMIT 10
            """
        )
    ).fetchall()
    if dupes:
        detay = ", ".join(f"{h[:12]}...x{n}" for h, n in dupes)
        raise RuntimeError(
            "photos.content_hash uzerinde UNIQUE index EKLENEMEDI: halihazirda "
            f"{len(dupes)} (ilk 10 gosteriliyor) yinelenen icerik grubu var -> {detay}. "
            "Cozum: her gruptan yalnizca birini birakip digerlerini "
            "DELETE /photos/{id} ile silin (dosya + Qdrant noktalari da "
            "temizlensin diye API uzerinden), sonra bu migration'i tekrar calistirin."
        )

    op.create_index(_NEW_INDEX, "photos", ["content_hash"], unique=True)
    op.drop_index(_OLD_INDEX, table_name="photos")


def downgrade() -> None:
    op.create_index(_OLD_INDEX, "photos", ["content_hash"], unique=False)
    op.drop_index(_NEW_INDEX, table_name="photos")
