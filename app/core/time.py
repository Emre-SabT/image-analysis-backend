"""UTC zaman damgasi yardimcilari.

DB'deki TUM DateTime kolonlari 'timestamp WITHOUT time zone' (naive) -
SQLAlchemy bunlari HER ZAMAN zaman dilimi bilgisi OLMADAN geri okur,
Python tarafinda ne yazilirsa yazilsin (bkz. db/models.py). Uygulama
genelinde bu naive degerlerin HER ZAMAN UTC oldugu KONVANSIYONU var
(hepsi `datetime.utcnow()` ile yaziliyor).

Sorun: naive bir datetime'da `.isoformat()` HICBIR zaman dilimi eki
(`Z` / `+00:00`) EKLEMEZ - RFC3339 UYUMLU DEGIL. JavaScript boyle bir
string'i (`new Date("2026-08-27T13:28:22")`) LOKAL SAAT olarak yorumlar,
UTC olarak DEGIL. Turkiye (UTC+3) icin bu, GERCEKTE "az once" olan bir
olayin "3 saat once" gibi gorunmesi demek - tam da kullanicinin bildirdigi
sapma (bkz. Son Etkinlik / En cok gorulen kisiler kullanici geri bildirimi).

Bu modul, JSON'a yazilmadan hemen once naive UTC degere ACIKCA +00:00
ekleyerek dogru bir RFC3339 string uretir - `to_iso_utc` HER `.isoformat()`
cagrisinin YERINE kullanilmalidir (bkz. cagrilan yerler).
"""

from datetime import datetime, timezone


def to_iso_utc(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()
