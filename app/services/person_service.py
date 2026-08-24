"""Kume etiketleme, kisi birlestirme, yuz yeniden atama - Bolum 13.1 uc noktalarinin is mantigi.

Nightly clustering (clustering_service.py) toplu/batch tarafi; burasi insan
etkilesimiyle tetiklenen (HITL) tekil islemleri kapsar (B2/B4, Bolum 10.2).
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from qdrant_client.models import PointStruct
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import identity_locks, qdrant
from app.db.models import Cluster, ClusterConstraint, Face, Person, Photo

logger = logging.getLogger("photoai.person_service")


def _face_vectors(face_ids: list[uuid.UUID]) -> dict[str, list[float]]:
    if not face_ids:
        return {}
    records = qdrant.client.retrieve(
        collection_name=qdrant.FACES_COLLECTION,
        ids=[str(fid) for fid in face_ids],
        with_vectors=True,
    )
    return {record.id: record.vector for record in records}


def _upsert_identity_centroid(db: Session, kind: str, identity_id: uuid.UUID) -> dict | None:
    """Bir kimligin (kisi ya da isimsiz kume) merkezini PG'de (persons.
    centroid/clusters.centroid - PR-C'den beri OTORITER) yeniden hesaplayip
    yazar; commit SONRASI cagiranin Qdrant'a (GECIS DONEMI ayna) uygulamasi
    icin bir dict doner.

    KATMAN 2 (bkz. PR-D tasarim notu): KENDI KISA transaction'inda calisir -
    cagiranin (label_cluster/merge_identities/reassign_face/delete_photo)
    ana transaction'i HER ZAMAN bundan ONCE commit etmis olur (bkz. 4
    cagri noktasinin da bu deseni izledigi dogrulamasi).

    KRITIK (revizyon turunde duzeltildi): face_ids parametre OLARAK
    ALINMAZ - kilit ALINDIKTAN SONRA, AYNI transaction icinde Face
    tablosundan TAZE sorgulanir. Eger face_ids CAGIRANDAN parametre olarak
    gelseydi (katman 1'in commit'i ile bu fonksiyonun kendi kilidi arasindaki
    pencerede), bir worker (_assign_or_bucket) ayni kimlige YENI bir yuz
    ekleyip DOGRU centroid'i commit edebilir - bu fonksiyon sonra ESKI
    (parametre olarak gelen) face_ids'ten hesapladigi centroid'i bunun
    UZERINE yazardi - klasik lost-update, tam da bu oturumun konusu.

    Bu TAZE-sorgu guvenlidir CUNKU: Face.person_id/cluster_id'yi degistiren
    HER yazar (worker'in _assign_or_bucket'i + bu dosyadaki merge/reassign/
    label/delete_photo) ONCE lock_identities alir - kilit alindigi an, o
    kimlige dokunan TUM onceki yazarlar ya commit etmis ya bu kilidi
    bekliyordur; taze SELECT (asagida) HER ZAMAN dogru, guncel uyeligi gorur.
    """
    identity_locks.lock_identities(db, [(kind, identity_id)])

    if kind == "person":
        face_ids = [f.id for f in db.query(Face).filter(Face.person_id == identity_id).all()]
    else:
        face_ids = [f.id for f in db.query(Face).filter(Face.cluster_id == identity_id).all()]

    vectors = _face_vectors(face_ids)
    if not vectors:
        db.commit()
        return None

    centroid = np.array(list(vectors.values())).mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    new_count = len(vectors)

    table = "persons" if kind == "person" else "clusters"
    count_col = "face_count" if kind == "person" else "size"
    # Kimlik "hala aktif" mi kontrolu (revizyon turu - Madde 6/9 bulgusu,
    # face_service._apply_assignment_locked'daki AYNI mantik): bu fonksiyon
    # commit SONRASI cagrildigi icin (Katman 2), Katman 1'in commit'i ile
    # buradaki kilit arasinda BASKA bir islem bu kimligi retire etmis
    # olabilir. WHERE'e eklenen kosul, retire edilmis bir satiri SESSIZCE
    # canlandirmaz - 0 satir etkilenirse asagida no-op donulur (bu 4
    # cagiran da senkron/istek-yaniti - worker'daki gibi bir requeue
    # mekanizmasi yok; identity zaten retire olduysa buraya yazmamak zaten
    # DOGRU sonuc, ayrica bir hata firlatmaya gerek yok).
    liveness_clause = "AND deleted_at IS NULL" if kind == "person" else "AND status = 'unlabeled'"
    result = db.execute(
        text(
            f"UPDATE {table} SET centroid = :c, {count_col} = :n, "
            f"centroid_updated_at = now() WHERE id = :id {liveness_clause}"
        ),
        {"c": centroid.astype(np.float32).tobytes(), "n": new_count, "id": identity_id},
    )
    db.commit()

    if result.rowcount == 0:
        # Kimlik artik aktif degil - centroid BILINCLI OLARAK yazilmadi.
        # Sessiz kalmasin - ileride teshis edilebilir olsun (bkz. onay
        # sonrasi eklenen log notu).
        logger.info(
            "_upsert_identity_centroid: kimlik artik aktif degil, centroid "
            "yazilmadi (kind=%s id=%s)", kind, identity_id,
        )
        return None

    return {
        "identity_id": str(identity_id),
        "centroid": centroid.tolist(),
        "payload": {"kind": kind, "id": str(identity_id), "face_count": new_count},
    }


def _recompute_or_delete_person(db: Session, person_id: uuid.UUID) -> bool:
    """Bir yuz kisiden ayrildiktan sonra PG tarafini gunceller.

    IKI DAL, IKI FARKLI SIRA KURALI (Aşama 1 duzeltmesi - PG >= Qdrant
    invariant'i; fonksiyon-seviyesi "yikici/yaratici" siniflandirmasi bu
    ayrimi ONCEDEN gizliyordu, iki dal artik AYRI ele aliniyor):

      - BOS KALIRSA (YIKICI): Qdrant'tan HEMEN siler - bu fonksiyon
        CAGIRANIN commit'inden ONCE cagrilmalidir. Yikici islemlerde
        Qdrant-once dogru yondur: commit basarisiz olup PG rollback olursa
        Person satiri geri gelir ama Qdrant zaten temizdir - guvenli yon
        (PG'de "fazla" veri kalir, bu onarilabilir).
      - UYE KALIRSA (YARATICI - centroid YENIDEN HESAPLANIYOR): SADECE PG'yi
        (face_count) gunceller, Qdrant'A DOKUNMAZ. Centroid'in Qdrant'a
        yazilmasi CAGIRANIN sorumlulugundadir ve MUTLAKA commit'ten SONRA,
        GUNCEL (commit sonrasi yeniden sorgulanmis) uyelikle yapilmalidir -
        bu bir "guncelleme" (yaratici) islemidir, PG-once kurali gecerlidir.
        Commit'ten once yazilirsa ve PG rollback olursa, Qdrant'ta ARTIK
        GECERLI OLMAYAN bir uyelige gore hesaplanmis yanlis bir centroid
        KALICI kalirdi.

    Kritik: bu adim olmazsa, yanlis atanmis bir yuzu klasorden cikarsaniz bile
    o yuzun katkisi kisinin merkezinde kalir - yani duzeltmeye calistiginiz
    kirlilik devam eder.

    Donus: True ise cagiran KENDI commit'inden SONRA bu person_id icin
    _upsert_identity_centroid("person", person_id, GUNCEL uye face_id'leri)
    cagirmalidir (bkz. caller'lar: reassign_face, photo_service.delete_photo).
    False ise kisi silinmis (Qdrant zaten temizlendi), baska bir sey
    yapilmasina gerek yok.
    """
    remaining = db.query(Face).filter(Face.person_id == person_id).all()
    if not remaining:
        qdrant.client.delete(
            collection_name=qdrant.IDENTITY_POOL_COLLECTION, points_selector=[str(person_id)]
        )
        db.query(Person).filter(Person.id == person_id).delete(synchronize_session=False)
        return False

    db.query(Person).filter(Person.id == person_id).update({"face_count": len(remaining)})
    return True


def _recompute_or_delete_cluster(db: Session, cluster_id: uuid.UUID) -> bool:
    """Ayni iki-dal/iki-sira kurali (bkz. _recompute_or_delete_person
    docstring'i) - kume icin. Donus: True ise cagiran commit SONRASI bu
    cluster_id icin centroid'i (GUNCEL uyelikle) yeniden yazmalidir."""
    remaining = db.query(Face).filter(Face.cluster_id == cluster_id).all()
    if not remaining:
        qdrant.client.delete(collection_name=qdrant.IDENTITY_POOL_COLLECTION, points_selector=[str(cluster_id)])
        db.query(Cluster).filter(Cluster.id == cluster_id).delete(synchronize_session=False)
        return False

    db.query(Cluster).filter(Cluster.id == cluster_id).update({"size": len(remaining)})
    return True


def list_clusters(db: Session, status: str = "unlabeled", sample_size: int = 5) -> list[dict]:
    """GET /clusters?status=unlabeled - isimlendirme bekleyen kumeler, ornek yuzlerle."""
    clusters = (
        db.query(Cluster)
        .filter(Cluster.status == status)
        .order_by(Cluster.created_at.desc())
        .all()
    )
    result = []
    for cluster in clusters:
        faces = db.query(Face).filter(Face.cluster_id == cluster.id).limit(sample_size).all()
        result.append(
            {
                "cluster_id": str(cluster.id),
                "status": cluster.status,
                "size": cluster.size,
                "created_at": cluster.created_at.isoformat() if cluster.created_at else None,
                "sample_faces": [
                    {
                        "face_id": str(f.id),
                        "photo_id": str(f.photo_id),
                        "crop_path": f.crop_path,
                        "det_confidence": f.det_confidence,
                    }
                    for f in faces
                ],
            }
        )
    return result


def list_persons(db: Session) -> list[dict]:
    """GET /persons - isimlendirilmis (aktif) tum kisiler, birer ornek yuzle ("klasor kapagi")."""
    persons = (
        db.query(Person)
        .filter(Person.deleted_at.is_(None))
        .order_by(Person.created_at.desc())
        .all()
    )
    result = []
    for person in persons:
        sample = db.query(Face).filter(Face.person_id == person.id).first()
        result.append(
            {
                "person_id": str(person.id),
                "display_name": person.display_name,
                "face_count": person.face_count,
                "created_at": person.created_at.isoformat() if person.created_at else None,
                "sample_face": (
                    {"face_id": str(sample.id), "photo_id": str(sample.photo_id)}
                    if sample
                    else None
                ),
            }
        )
    return result


def get_photos_for_person(db: Session, person_id: uuid.UUID):
    """GET /persons/{id}/photos - bir kisinin yer aldigi tum fotograflar (Bolum 13.1)."""
    return (
        db.query(Photo)
        .join(Face, Face.photo_id == Photo.id)
        .filter(Face.person_id == person_id)
        .distinct()
        .order_by(Photo.created_at.desc())
        .all()
    )


def label_cluster(db: Session, cluster_id: uuid.UUID, display_name: str, created_by_user_id: uuid.UUID | None) -> Person:
    """POST /clusters/{id}/label (B2, Bolum 10.2) - kumeye kisi adi atar, tum uyelere yayar."""
    # PR-D: cluster HENUZ "unlabeled" iken bir worker (_assign_or_bucket) ona
    # yeni bir yuz ekliyor olabilir - erken kilit, bu okuma-sonra-donustur
    # adimini o yarisa karsi korur (worker'in KENDI lock_identities cagrisi
    # bu satiri zaten kilitli bulup bekler/commit sonrasi devam eder).
    identity_locks.lock_identities(db, [("cluster", cluster_id)])

    cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
    if cluster is None:
        raise ValueError("Kume bulunamadi")
    if cluster.status != "unlabeled":
        raise ValueError(f"Kume zaten '{cluster.status}' durumunda")

    faces = db.query(Face).filter(Face.cluster_id == cluster.id).all()
    face_ids = [f.id for f in faces]

    person = Person(
        display_name=display_name,
        cluster_id=cluster.id,
        face_count=len(face_ids),
        created_by_user_id=created_by_user_id,
        created_at=datetime.utcnow(),
    )
    db.add(person)
    db.flush()

    db.query(Face).filter(Face.id.in_(face_ids)).update(
        {"person_id": person.id, "assigned_by": "human"}, synchronize_session=False
    )
    cluster.status = "labeled"
    cluster.centroid_updated_at = datetime.utcnow()

    # Kume artik isimlendirildi; kimlik havuzundaki eski "kind=cluster" kaydi
    # gereksiz - yerine asagida (commit SONRASI) "kind=person" kaydi
    # yazilacak. YIKICI islem (Aşama 0 duzeltmesi - PG-once DEGIL,
    # Qdrant-once): PG commit'inden ONCE silinir - aksi halde commit
    # basarisiz olursa PG'de kume "labeled" durumunda kalirken Qdrant'ta
    # hala "kind=cluster" aktif bir kayit olarak durur, _assign_or_bucket
    # yeni yuzleri yanlislikla bu artik-gecersiz kumeye yonlendirebilir.
    qdrant.client.delete(collection_name=qdrant.IDENTITY_POOL_COLLECTION, points_selector=[str(cluster.id)])

    db.commit()
    db.refresh(person)

    # PR-D, KATMAN 2: KENDI kilidini alir, uyeligi TAZE sorgular (yukaridaki
    # face_ids DEGIL - bkz. _upsert_identity_centroid docstring'i), PG
    # centroid'i yazar. Donen dict, commit SONRASI (Faz 3) Qdrant dual-write
    # icin.
    op = _upsert_identity_centroid(db, "person", person.id)
    if op is not None:
        qdrant.client.upsert(
            collection_name=qdrant.IDENTITY_POOL_COLLECTION,
            points=[PointStruct(id=op["identity_id"], vector=op["centroid"], payload=op["payload"])],
        )
    qdrant.client.set_payload(
        collection_name=qdrant.FACES_COLLECTION,
        payload={"person_id": str(person.id)},
        points=[str(fid) for fid in face_ids],
    )
    return person


def _faces_of_identity(db: Session, kind: str, identity_id: uuid.UUID) -> list[Face]:
    if kind == "person":
        return db.query(Face).filter(Face.person_id == identity_id).all()
    return db.query(Face).filter(Face.cluster_id == identity_id).all()


def _is_merge_blocked(
    db: Session, target_kind: str, target_id: uuid.UUID, source_kind: str, source_id: uuid.UUID
) -> bool:
    """target ve source arasinda bir cannot_link kisiti var mi - reject_merge
    ile "bunlar ayni kisi DEGIL" denmis bir cift, merge_identities ile
    (dogrudan bir cagriyla ya da bayat/eski bir oneriyi kabul ederek) YINE DE
    birlestirilemez.

    NEDEN (bayat oneri sorununun bir parcasi): reject_merge'un yazdigi
    cannot_link kisiti ONCEDEN SADECE clustering_service.suggest_merges'in
    kendi filtresinde (_blocked_pairs/_is_blocked) kontrol ediliyordu - yani
    SADECE yeni oneri URETILMESINI engelliyordu. merge_identities'in KENDISI
    bu kisidi hic sormuyordu - bir admin bir cifti reddettikten SONRA, elinde
    kalan (artik bayat) bir oneriyi yine de "kabul et" ile birlestirebiliyordu.

    cannot_link kisitlari YUZ ciftleri uzerinde tutuluyor (ClusterConstraint) -
    bu fonksiyon her kisitin iki yuzunun SU ANKI (kisit yaratildigi andaki
    DEGIL) kimligini cozup target/source ciftiyle eslesip eslesmedigine bakar -
    clustering_service._blocked_pairs ile AYNI mantik, TEK bir cifte
    daraltilmis (tum kisitlari degil, sadece bu ikisini ilgilendirenleri tarar).
    """
    target_pair = (target_kind, target_id)
    source_pair = (source_kind, source_id)
    constraints = db.query(ClusterConstraint).filter(ClusterConstraint.type == "cannot_link").all()
    for c in constraints:
        fa = db.query(Face).filter(Face.id == c.face_id_a).first()
        fb = db.query(Face).filter(Face.id == c.face_id_b).first()
        if not fa or not fb:
            continue
        ia = ("person", fa.person_id) if fa.person_id else ("cluster", fa.cluster_id)
        ib = ("person", fb.person_id) if fb.person_id else ("cluster", fb.cluster_id)
        if {ia, ib} == {target_pair, source_pair}:
            return True
    return False


def reject_merge(db: Session, identities: list[dict], created_by_user_id: uuid.UUID | None = None) -> dict:
    """"Bunlar ayni kisi DEGIL" bilgisini kalici kisit olarak kaydeder.

    Verilen kimliklerin her ikili kombinasyonu icin temsili birer yuz arasina
    cannot_link yazilir. clustering_service._blocked_pairs bunlari okur, yani
    ayni yanlis oneri bir daha uretilmez (B4/B5, Bolum 10.2 geri besleme).
    """
    reps: list[tuple[str, uuid.UUID, Face]] = []
    for item in identities:
        kind, ident = item["kind"], item["id"]
        face = _faces_of_identity(db, kind, ident)
        if face:
            reps.append((kind, ident, face[0]))

    if len(reps) < 2:
        raise ValueError("En az iki klasor gerekli")

    added = 0
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            db.add(
                ClusterConstraint(
                    face_id_a=reps[i][2].id,
                    face_id_b=reps[j][2].id,
                    type="cannot_link",
                    created_by_user_id=created_by_user_id,
                )
            )
            added += 1
    db.commit()
    return {"constraints_added": added}


def delete_identity(db: Session, kind: str, identity_id: uuid.UUID) -> dict:
    """Bir klasoru (kisi ya da isimsiz kume) ve icindeki TUM yuz kayitlarini
    kalici olarak siler.

    Kullanim amaci iki yonlu:
      1. Yanlis tespit (false positive) temizligi - YuNet bazen yuz olmayan bir
         bolgeye yuksek guven verebiliyor; hicbir esik bunu guvenle elemiyor
         (bkz. Bolum 8.1 esik notu), o yuzden insan duzeltmesi gerekiyor.
      2. FR-13 / KVKK: bir kisiye ait tum verinin talep uzerine silinmesi.

    Silinenler: face_suggestions + cluster_constraints (FK bagimliliklari),
    yuz kirpim dosyalari, Qdrant'taki yuz vektorleri, kimlik havuzu kaydi,
    Face satirlari ve klasorun kendisi.

    Silinmeyenler: FOTOGRAFLAR. Ayni fotografta gecerli baska yuzler
    olabilecegi icin photos tablosuna ve dosyalarina dokunulmaz.
    """
    if kind not in ("person", "cluster"):
        raise ValueError("Gecersiz klasor turu")

    if kind == "person":
        identity = db.query(Person).filter(Person.id == identity_id).first()
    else:
        identity = db.query(Cluster).filter(Cluster.id == identity_id).first()
    if identity is None:
        raise ValueError("Klasor bulunamadi")

    faces = _faces_of_identity(db, kind, identity_id)
    face_ids = [f.id for f in faces]

    if face_ids:
        # FK bagimliliklari once temizlenmeli.
        db.query(ClusterConstraint).filter(
            (ClusterConstraint.face_id_a.in_(face_ids))
            | (ClusterConstraint.face_id_b.in_(face_ids))
        ).delete(synchronize_session=False)

        for face in faces:
            crop = Path(face.crop_path)
            if crop.exists():
                try:
                    crop.unlink()
                except OSError:
                    pass  # dosya silinemezse DB temizligi yine de surmeli

        qdrant.client.delete(
            collection_name=qdrant.FACES_COLLECTION,
            points_selector=[str(fid) for fid in face_ids],
        )
        db.query(Face).filter(Face.id.in_(face_ids)).delete(synchronize_session=False)

    # Kisi silinirken, o kisinin dogdugu isimsiz kume kaydi da artik anlamsiz.
    linked_cluster_id = getattr(identity, "cluster_id", None) if kind == "person" else None

    if kind == "person":
        db.query(Person).filter(Person.id == identity_id).delete(synchronize_session=False)
        if linked_cluster_id is not None:
            db.query(Cluster).filter(Cluster.id == linked_cluster_id).delete(synchronize_session=False)
    else:
        db.query(Cluster).filter(Cluster.id == identity_id).delete(synchronize_session=False)

    # KVKK/FR-13 - Aşama 0 duzeltmesi (PG >= Qdrant invariant'i): identity_pool
    # kaydi PG commit'inden ONCE silinir. ONCEKI siralamada (PG-once) commit
    # basarisiz olsa bile Qdrant'taki biyometrik centroid KALICI olarak
    # sizabiliyordu - PG "silindi" derken Qdrant hala tasiyordu, hicbir
    # retry/alarm yoktu (KVKK acigi). Simdi: bu satir patlarsa fonksiyonun
    # geri kalani (asagidaki commit dahil) HIC calismaz, PG rollback olur -
    # kisi PG'de GERI GELIR (gorunur, admin tekrar silebilir - Qdrant'tan
    # zaten-silinmis bir noktayi tekrar silmek hata vermez) ama Qdrant'ta
    # ASLA "PG'nin sildigini soyledigi ama Qdrant'in tuttugu" bir veri
    # KALAMAZ - sizinti yapisal olarak imkansiz hale geldi.
    #
    # KALAN RISK (bu tek degisiklikle KAPANMAZ - Aşama 3 reconciliation'in
    # cift yonlu taramasina BAGIMLI): Qdrant silme BASARILI olup PG commit
    # (nadiren - deadlock, baglanti kopmasi) PATLARSA, kisi PG'de geri gelir
    # AMA centroid'i Qdrant'tan GITMIS olur. Bu GORUNMEZ bir durumdur -
    # kullaniciya hicbir hata gostermez, kisi 'Kisiler' ekraninda hala var
    # gorunur (PG rollback sayesinde). O aralikta yuklenen fotograflar bu
    # kisiyi ARTIK BULAMAZ (identity_pool'da karsiligi yok) ve
    # YANLISLIKLA yeni bir isimsiz kume acar - kisi sessizce "bolunur".
    # Bu durum, Aşama 3 reconciliation'in "PG satiri var ama Qdrant'ta
    # karsiligi yok" yonlu taramasi calisip merkezi yeniden yazana kadar
    # SESSIZCE kalir; bu PR'in kapsaminda COZULMUYOR, sadece belgeleniyor.
    pool_ids = [str(identity_id)] + ([str(linked_cluster_id)] if linked_cluster_id else [])
    qdrant.client.delete(collection_name=qdrant.IDENTITY_POOL_COLLECTION, points_selector=pool_ids)

    db.commit()

    return {"deleted_kind": kind, "deleted_id": str(identity_id), "deleted_faces": len(face_ids)}


def merge_identities(
    db: Session,
    target_kind: str,
    target_id: uuid.UUID,
    source_kind: str,
    source_id: uuid.UUID,
    created_by_user_id: uuid.UUID | None = None,
) -> dict:
    """İki kimligi (kisi ya da isimsiz kume, herhangi bir kombinasyon) birlestirir
    (B4, Bolum 10.2 - insan geri bildirimi). Sonuc daima target_kind'in turunde
    kalir: target bir kisi ise sonuc isimli kisi, target bir kume ise sonuc
    hala isimsiz kume olarak kalir (kullanici sonra isimlendirebilir)."""
    if target_kind == source_kind and target_id == source_id:
        raise ValueError("Bir klasor kendisiyle birlestirilemez")
    if target_kind not in ("person", "cluster") or source_kind not in ("person", "cluster"):
        raise ValueError("Gecersiz klasor turu")

    # Bayat oneri duzeltmesi (Seviye 1): reject_merge ile "ayni kisi DEGIL"
    # denmis bir cift, dogrudan cagriyla ya da eski/bayat bir oneriyi kabul
    # ederek YINE DE birlestirilemez. Kilit almadan ONCE, ucuz bir kontrol.
    if _is_merge_blocked(db, target_kind, target_id, source_kind, source_id):
        raise ValueError(
            "Bu iki klasor daha once 'ayni kisi degil' olarak isaretlenmis "
            "(reddedilmis bir birlestirme onerisi) - birlestirilemez."
        )

    # PR-D, KATMAN 1: iki kimlik (target+source) TEK cagrida, ic siralamayla
    # kilitlenir - iki merge_identities cagrisi TERS sirada (biri A->B, digeri
    # B->A) ayni ciftle cakisirsa DEADLOCK olusmasin (bkz. identity_locks.py
    # tasarim notu). Bu, asagidaki target face_count/source silme
    # yazimlarini bir worker'in eszamanli artimli guncellemesine karsi korur.
    identity_locks.lock_identities(db, [(target_kind, target_id), (source_kind, source_id)])

    target_faces = _faces_of_identity(db, target_kind, target_id)
    source_faces = _faces_of_identity(db, source_kind, source_id)
    if not target_faces or not source_faces:
        # Bayat oneri duzeltmesi (Seviye 1): mesaj netlestirildi - bu hata,
        # tipik olarak eski bir "birlestirme onerisi" listesindeki bir
        # klasorun ARADA baska bir islemle (merge/silme) zaten yok olmasindan
        # kaynaklanir.
        raise ValueError(
            "Klasor(ler) artik mevcut degil ya da bos - muhtemelen daha once "
            "birlestirilmis/silinmis. Oneri listesini yenileyip tekrar deneyin."
        )

    # Insan geri bildirimi (Bolum 8.3): merge -> must-link kisiti (temsili bir cift).
    db.add(
        ClusterConstraint(
            face_id_a=target_faces[0].id,
            face_id_b=source_faces[0].id,
            type="must_link",
            created_by_user_id=created_by_user_id,
        )
    )

    source_face_ids = [f.id for f in source_faces]
    if target_kind == "person":
        db.query(Face).filter(Face.id.in_(source_face_ids)).update(
            {"person_id": target_id, "cluster_id": None, "assigned_by": "human"}, synchronize_session=False
        )
    else:
        db.query(Face).filter(Face.id.in_(source_face_ids)).update(
            {"cluster_id": target_id, "person_id": None, "assigned_by": "human"}, synchronize_session=False
        )

    if source_kind == "person":
        source = db.query(Person).filter(Person.id == source_id).first()
        if source.cluster_id is not None:
            db.query(Cluster).filter(Cluster.id == source.cluster_id).update({"status": "merged"})
        source.deleted_at = datetime.utcnow()
        # PR-D duzeltmesi ("anna" vakasi): soft-delete edilen source'un
        # face_count'u ONCEDEN sifirlanmiyordu, eski (yanlis) degerinde
        # donuk kaliyordu. Ayni transaction'da, yukaridaki (Katman 1) kilit
        # altinda sifirlanir.
        db.query(Person).filter(Person.id == source_id).update({"face_count": 0})
    else:
        db.query(Cluster).filter(Cluster.id == source_id).delete(synchronize_session=False)

    all_face_ids = [f.id for f in target_faces] + source_face_ids

    if target_kind == "person":
        db.query(Person).filter(Person.id == target_id).update({"face_count": len(all_face_ids)})
    else:
        db.query(Cluster).filter(Cluster.id == target_id).update({"size": len(all_face_ids)})

    # source'un identity_pool kaydi artik gecersiz (uyeleri target'a tasindi).
    # YIKICI islem (Aşama 0 duzeltmesi - PG-once DEGIL, Qdrant-once): commit'ten
    # ONCE silinir - aksi halde bu adim commit sonrasi basarisiz olursa,
    # Qdrant'ta soft-delete edilmis (source.deleted_at set) bir kisiye ait
    # AKTIF bir identity_pool kaydi kalir; _assign_or_bucket yeni yuzleri
    # sessizce bu "silinmis" kisiye atayabilir.
    qdrant.client.delete(collection_name=qdrant.IDENTITY_POOL_COLLECTION, points_selector=[str(source_id)])

    db.commit()

    # PR-D, KATMAN 2: KENDI kilidini alir, uyeligi TAZE sorgular (yukaridaki
    # all_face_ids DEGIL), PG centroid'i yazar.
    op = _upsert_identity_centroid(db, target_kind, target_id)
    if op is not None:
        qdrant.client.upsert(
            collection_name=qdrant.IDENTITY_POOL_COLLECTION,
            points=[PointStruct(id=op["identity_id"], vector=op["centroid"], payload=op["payload"])],
        )

    if target_kind == "person":
        qdrant.client.set_payload(
            collection_name=qdrant.FACES_COLLECTION,
            payload={"person_id": str(target_id), "cluster_id": None},
            points=[str(fid) for fid in source_face_ids],
        )
    else:
        qdrant.client.set_payload(
            collection_name=qdrant.FACES_COLLECTION,
            payload={"cluster_id": str(target_id), "person_id": None},
            points=[str(fid) for fid in source_face_ids],
        )

    return {"kind": target_kind, "id": str(target_id), "face_count": len(all_face_ids)}


def reassign_face(
    db: Session,
    face_id: uuid.UUID,
    person_id: uuid.UUID | None,
    created_by_user_id: uuid.UUID | None = None,
) -> Face:
    """POST /faces/{id}/reassign - yanlis atanmis bir yuzu duzeltir (B4, split).

    person_id verilirse : yuz o kisiye tasinir.
    person_id None ise  : yuz mevcut klasorunden ayrilir ve KENDI yeni isimsiz
        klasorunu acar - boylece 'Kisiler' ekraninda gorunur kalir ve dogru
        kimligi sonradan atanabilir (kimliksiz havuzda kaybolmaz).

    Her iki durumda da ESKI kimligin merkezi yeniden hesaplanir; aksi halde
    yanlis yuzun katkisi eski kisinin/klasorun merkezinde kalmaya devam ederdi.
    """
    face = db.query(Face).filter(Face.id == face_id).first()
    if face is None:
        raise ValueError("Yuz bulunamadi")

    if person_id is not None:
        target = db.query(Person).filter(Person.id == person_id, Person.deleted_at.is_(None)).first()
        if target is None:
            raise ValueError("Kisi bulunamadi")
        if face.person_id == person_id:
            raise ValueError("Yuz zaten bu kisiye ait")

    old_person_id = face.person_id
    old_cluster_id = face.cluster_id

    # PR-D, KATMAN 1: eski + yeni (ikisi de ONCEDEN VAR olan) kimlikler TEK
    # cagrida kilitlenir - deadlock-guvenli sira icin (bkz. identity_locks.py
    # tasarim notu, merge_identities'teki AYNI desen). Yeni acilan kume
    # (person_id None dalinda) kilit GEREKTIRMEZ - satiri henuz yok, kimse
    # referans veremez.
    identities_to_lock = []
    if old_person_id is not None:
        identities_to_lock.append(("person", old_person_id))
    if old_cluster_id is not None:
        identities_to_lock.append(("cluster", old_cluster_id))
    if person_id is not None:
        identities_to_lock.append(("person", person_id))
    identity_locks.lock_identities(db, identities_to_lock)

    # Insan geri bildirimi (Bolum 8.3): split -> cannot-link kisiti. Eski
    # klasorde kalan temsili bir uyeyle "bunlar ayni kisi DEGIL" kaydi dusulur.
    # (Kisiden cikarma da dahil - onceki surumde yalnizca kume icin yaziliyordu.)
    if old_person_id is not None:
        remaining = (
            db.query(Face).filter(Face.person_id == old_person_id, Face.id != face.id).first()
        )
    elif old_cluster_id is not None:
        remaining = (
            db.query(Face).filter(Face.cluster_id == old_cluster_id, Face.id != face.id).first()
        )
    else:
        remaining = None

    if remaining is not None:
        db.add(
            ClusterConstraint(
                face_id_a=face.id,
                face_id_b=remaining.id,
                type="cannot_link",
                created_by_user_id=created_by_user_id,
            )
        )

    new_cluster_id = None
    if person_id is not None:
        face.person_id = person_id
        face.cluster_id = None
    else:
        new_cluster = Cluster(status="unlabeled", size=1, created_at=datetime.utcnow())
        db.add(new_cluster)
        db.flush()
        new_cluster_id = new_cluster.id
        face.person_id = None
        face.cluster_id = new_cluster_id
    face.assigned_by = "human"
    db.flush()

    # Yeni kimligin face_count'u (SADECE mevcut bir kisiye tasiniyorsa - yeni
    # kume dalinda zaten Cluster(size=1,...) ile dogru doguyor) - Katman 1'in
    # ERKEN kilidi altinda, ayni commit'e dahil (eskiden ayri bir ikinci
    # commit vardi, artik gerek yok).
    if person_id is not None:
        new_member_count = db.query(Face).filter(Face.person_id == person_id).count()
        db.query(Person).filter(Person.id == person_id).update({"face_count": new_member_count})

    # Eski kimligin merkezini/sayacini duzelt (bossa sil, YIKICI - Qdrant
    # HEMEN silinir; uye kaldiysa SADECE PG guncellenir, centroid asagida
    # commit SONRASI GUNCEL uyelikle yazilir - bkz.
    # _recompute_or_delete_person/cluster docstring'i, Aşama 1 duzeltmesi).
    old_person_needs_recompute = False
    old_cluster_needs_recompute = False
    if old_person_id is not None:
        old_person_needs_recompute = _recompute_or_delete_person(db, old_person_id)
    if old_cluster_id is not None and old_cluster_id != new_cluster_id:
        old_cluster_needs_recompute = _recompute_or_delete_cluster(db, old_cluster_id)

    db.commit()
    db.refresh(face)

    # PR-D, KATMAN 2: her etkilenen kimlik icin KENDI kilidini alir, uyeligi
    # TAZE sorgular, PG centroid'i yazar. Donen dict'ler commit SONRASI
    # (Faz 3) Qdrant dual-write icin biriktirilir.
    pending_identity_ops = []
    if old_person_needs_recompute:
        op = _upsert_identity_centroid(db, "person", old_person_id)
        if op is not None:
            pending_identity_ops.append(op)
    if old_cluster_needs_recompute:
        op = _upsert_identity_centroid(db, "cluster", old_cluster_id)
        if op is not None:
            pending_identity_ops.append(op)

    qdrant.client.set_payload(
        collection_name=qdrant.FACES_COLLECTION,
        payload={
            "person_id": str(person_id) if person_id else None,
            "cluster_id": str(new_cluster_id) if new_cluster_id else None,
        },
        points=[str(face.id)],
    )

    # Yeni kimligin merkezini yaz.
    if person_id is not None:
        op = _upsert_identity_centroid(db, "person", person_id)
    else:
        op = _upsert_identity_centroid(db, "cluster", new_cluster_id)
    if op is not None:
        pending_identity_ops.append(op)

    for op in pending_identity_ops:
        qdrant.client.upsert(
            collection_name=qdrant.IDENTITY_POOL_COLLECTION,
            points=[PointStruct(id=op["identity_id"], vector=op["centroid"], payload=op["payload"])],
        )

    return face
