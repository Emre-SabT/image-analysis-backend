"""Person/Cluster satirlarini DETERMINISTIK sirada kilitleyen tek yardimci
(identity centroid eszamanlilik calismasi). Amac: birden fazla kimligi
AYNI transaction'da guncelleyen akislar (merge_identities, reassign_face -
PR-D'de bu yardimciyi kullanacak sekilde guncellenecek; face_service.py'nin
tek-kimlikli artimli guncellemesi de - N=1 icin sira sorunu yok ama tutarlilik
icin - bu yardimciyi kullanir) DEADLOCK yaratmasin.

TASARIM KARARLARI (revizyon turlerinde tartisildi, TEKRAR ACMA):

  - Siralama anahtari SADECE id DEGIL, (kind, id) ciftinin STRING temsili.
    persons ve clusters AYRI tablolar; iki transaction'in (person X,
    cluster Y) ve (cluster Y, person X) sirasiyla kilit istemesi CEMBER
    BEKLEMEYE (deadlock) yol acar. SADECE id'ye gore siralamak bunu ayirt
    EDEMEZ - kind ONEKI olmadan capraz-tablo deterministik sira garantisi
    YOKTUR (farkli tablolardaki UUID'lerin sirali karsilastirilmasi,
    hangi TABLODAN geldiklerini bilmeden anlamli degildir).

  - Acik bir DONGUDE, tek-tek SIRALI `SELECT ... FOR UPDATE` cagrilir -
    tek bir `WHERE id = ANY(...) ORDER BY id FOR UPDATE` sorgusu YERINE.
    Sebep: Postgres'in LockRows dugumunun Sort'tan ONCE mi SONRA mi
    calistigi (yani kilit ACILIS sirasinin ORDER BY ciktisiyla birebir
    eslesip eslesmedigi) DOGRULANMADI - bu belirsizlikle deadlock-guvenligi
    riske atmaktansa, garantili-sirali ama biraz daha yavas (N kucuk
    oldugu icin ihmal edilebilir - genelde 1-2 kimlik) acik donguyu
    tercih ediyoruz.

  - Bloke OLUR (NOWAIT DEGIL): contention NADIR beklenir (farkli
    fotograflarin/isteklerin AYNI kimlige AYNI anda dusme ihtimali dusuk),
    kisa bir bekleme normal calismanin parcasi. lock_timeout ile UST SINIR
    konur (varsayilan 5000ms) - sizmis/unutulmus bir kilit yuzunden
    SONSUZA dek beklemeyi engeller; asilirsa Postgres 'lock_timeout'
    hatasi firlatir (psycopg2.errors.LockNotAvailable), cagiran tarafin
    MEVCUT hata/retry mekanizmasina (worker: jobs.fail(retry=True) ->
    backoff'la yeniden dener; HTTP: 500, kullanici tekrar dener) dusmesi
    beklenir - YENI bir hata sinifi degil.
"""

import uuid

import psycopg2.errors
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

_TABLE_BY_KIND = {"person": "persons", "cluster": "clusters"}


class IdentityLockTimeout(Exception):
    """lock_timeout asilinca firlatilir - baska bir islem ayni kimligi (ya
    da kimliklerden birini) O AN tutuyor. GERCEK bir hata DEGIL (PR-2'nin
    LockConflict'iyle AYNI SINIF - 'kaynak o an mesguldu' bilgisi, gercek
    bir mantik/veri hatasi degil).

    BILINCLI OLARAK burada (db katmani) dogrudan photo_service.LockConflict
    FIRLATILMIYOR: identity_locks.py db katmaninda, LockConflict services
    katmaninda (photo_service.py) tanimli - oradan import etmek CEMBER
    IMPORT yaratirdi (photo_service -> face_service -> identity_locks ->
    photo_service). Cagiran katman (bkz. photo_service.run_face_pipeline_job)
    bu exception'i YAKALAYIP kendi LockConflict'ine CEVIRIR - boylece
    worker/main.py'nin ZATEN VAR OLAN LockConflict islem hatti (cezasiz
    requeue + esik-asilinca-eskalasyon) hicbir DEGISIKLIK gerektirmeden
    yeniden kullanilir.
    """


class StaleIdentityDecision(Exception):
    """FAZ 1'de (bkz. face_service.py:_decide_assignment) karar verilen
    kimlik, FAZ 2'ye (kilit alindiktan, TAZE SELECT yapildiktan) gelindiginde
    ARTIK GECERLI DEGIL - soft-delete edilmis (persons.deleted_at) ya da
    isimlendirilmis/birlestirilmis (clusters.status != 'unlabeled'). Baska
    bir islem (admin merge/delete/label) FAZ 1 ile FAZ 2 arasinda araya
    girmis demektir - worker'in kararinin KENDISI o an dogruydu, ARADAN
    GECEN SURE onu GECERSIZ kildi.

    GERCEK bir hata DEGIL (IdentityLockTimeout ile AYNI SINIF - 'kaynak
    artik bu haliyle yok/degisti' bilgisi). AYNI GEREKCEYLE (cember import
    onlemek icin) burada degil, cagiran katmanda (photo_service.
    run_face_pipeline_job) LockConflict'e cevrilir - worker/main.py'nin
    cezasiz-requeue hatti FAZ 1'i BASTAN calistirir, yeni (guncel) bir
    karar verilir.
    """


def lock_identities(
    db: Session,
    identities: list[tuple[str, uuid.UUID]],
    lock_timeout_ms: int = 5000,
) -> None:
    """Verilen (kind, id) ciftlerini SIRALI sekilde `SELECT ... FOR UPDATE`
    ile kilitler.

    CAGIRANIN KENDI ACIK transaction'i icinde calisir - kilitler o
    transaction commit/rollback olana kadar tutulur. Bu fonksiyon SADECE
    kilitler, veri DONDURMEZ - cagiran, kilitledikten SONRA kendi SELECT'ini
    calistirip guncel degerleri okumali (iki adima bolunmus olmasi,
    "hangi kimlikler kilitlenecek" listesinin cagiran tarafindan ONCEDEN
    bilinmesi gereken durumlarda - ornegin merge_identities'in hem target
    hem source'u - daha acik bir sozlesme).
    """
    if not identities:
        return
    db.execute(text(f"SET LOCAL lock_timeout = '{lock_timeout_ms}ms'"))
    try:
        for kind, identity_id in sorted(identities, key=lambda t: (t[0], str(t[1]))):
            table = _TABLE_BY_KIND[kind]
            db.execute(text(f"SELECT id FROM {table} WHERE id = :id FOR UPDATE"), {"id": identity_id})
    except OperationalError as exc:
        if isinstance(getattr(exc, "orig", None), psycopg2.errors.LockNotAvailable):
            raise IdentityLockTimeout(
                f"Kimlik kilidi zaman asimina ugradi ({lock_timeout_ms}ms): {identities}"
            ) from exc
        raise
