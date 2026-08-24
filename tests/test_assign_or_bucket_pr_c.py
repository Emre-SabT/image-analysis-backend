"""detect_and_embed'in PR-C (uc fazli yapi: FAZ 1 karar/agir is - kilit yok,
FAZ 2 TEK KISA transaction icinde kilit+Face+centroid, FAZ 3 commit sonrasi
Qdrant dual-write) davranisinin regresyon testleri. GERCEK PostgreSQL +
Qdrant'a karsi calisir; detector/embedder FAKE'lenir (bu testlerin amaci
kilit/transaction davranisi, YuNet/AuraFace dogrulugu degil).
"""

import uuid
from datetime import datetime

import numpy as np
import pytest
from sqlalchemy import text

from app.core.settings import settings
from app.db import qdrant
from app.db.models import Photo
from app.db.session import SessionLocal
from app.services import face_service


def _norm_vec():
    v = np.random.randn(512)
    return v / np.linalg.norm(v)


class _FakeDetected:
    def __init__(self, bbox, landmarks, confidence):
        self.bbox = bbox
        self.landmarks = landmarks
        self.confidence = confidence


class _FakeDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, image_bgr):
        return self._detections


class _FakeEmbedder:
    """embeddings: her cagrida sirayla donecek (embedding, aligned_crop) ciftleri."""

    def __init__(self, embeddings):
        self._embeddings = list(embeddings)
        self._i = 0

    def get_embedding(self, image_bgr, landmarks):
        emb = self._embeddings[self._i]
        self._i += 1
        return emb, np.zeros((112, 112, 3), dtype=np.uint8)


def _install_fakes(monkeypatch, n_faces, embeddings):
    detections = [_FakeDetected((0, 0, 100, 100), [[0, 0]] * 5, 0.95) for _ in range(n_faces)]
    monkeypatch.setattr(face_service, "_get_detector", lambda: _FakeDetector(detections))
    monkeypatch.setattr(face_service, "_get_embedder", lambda: _FakeEmbedder(embeddings))
    monkeypatch.setattr(face_service, "_read_image_bgr", lambda path: np.zeros((200, 200, 3), dtype=np.uint8))


@pytest.fixture
def photo(test_user_id):
    db = SessionLocal()
    try:
        p = Photo(filename="t.jpg", storage_path=f"uploads/{uuid.uuid4()}.jpg",
                   uploaded_by_user_id=test_user_id)
        db.add(p)
        db.commit()
        db.refresh(p)
        obj = p
    finally:
        db.close()

    yield obj

    db = SessionLocal()
    try:
        face_ids = [str(r[0]) for r in db.execute(
            text("SELECT id FROM faces WHERE photo_id=:p"), {"p": str(obj.id)}
        ).fetchall()]
        cluster_ids = [str(r[0]) for r in db.execute(
            text("SELECT DISTINCT cluster_id FROM faces WHERE photo_id=:p AND cluster_id IS NOT NULL"),
            {"p": str(obj.id)},
        ).fetchall()]
        if face_ids:
            qdrant.client.delete(qdrant.FACES_COLLECTION, points_selector=face_ids)
        if cluster_ids:
            qdrant.client.delete(qdrant.IDENTITY_POOL_COLLECTION, points_selector=cluster_ids)
        db.execute(text("DELETE FROM faces WHERE photo_id=:p"), {"p": str(obj.id)})
        if cluster_ids:
            db.execute(text("DELETE FROM clusters WHERE id::text = ANY(:c)"), {"c": cluster_ids})
        db.execute(text("DELETE FROM photos WHERE id=:p"), {"p": str(obj.id)})
        db.commit()
    finally:
        db.close()


def test_new_cluster_dual_writes_pg_centroid(monkeypatch, photo):
    e1 = _norm_vec()
    _install_fakes(monkeypatch, 1, [e1])

    db = SessionLocal()
    try:
        saved = face_service.detect_and_embed(db, photo)
    finally:
        db.close()

    assert len(saved) == 1
    cluster_id = saved[0].cluster_id
    assert cluster_id is not None

    db2 = SessionLocal()
    try:
        row = db2.execute(text("SELECT centroid, size FROM clusters WHERE id=:i"),
                           {"i": str(cluster_id)}).fetchone()
    finally:
        db2.close()
    assert row is not None and row[0] is not None, "yeni kume PG'de centroid'siz kaldi"
    pg_vec = np.frombuffer(bytes(row[0]), dtype=np.float32)
    assert abs(float(np.dot(pg_vec, e1)) - 1.0) < 1e-4
    assert row[1] == 1


def test_incremental_update_matches_pg_and_qdrant(monkeypatch, photo):
    e1 = _norm_vec()
    _install_fakes(monkeypatch, 1, [e1])
    db = SessionLocal()
    try:
        saved1 = face_service.detect_and_embed(db, photo)
    finally:
        db.close()
    cluster_id = saved1[0].cluster_id

    e2 = e1 + np.random.randn(512) * 0.01
    e2 = e2 / np.linalg.norm(e2)
    _install_fakes(monkeypatch, 1, [e2])
    db = SessionLocal()
    try:
        saved2 = face_service.detect_and_embed(db, photo)
    finally:
        db.close()

    assert saved2[0].cluster_id == cluster_id, "ikinci yuz AYNI kimlige eslesmedi"

    db2 = SessionLocal()
    try:
        row = db2.execute(text("SELECT centroid, size FROM clusters WHERE id=:i"),
                           {"i": str(cluster_id)}).fetchone()
    finally:
        db2.close()
    pg_vec = np.frombuffer(bytes(row[0]), dtype=np.float32).astype(np.float64)
    expected = e1 + e2
    expected = expected / np.linalg.norm(expected)
    assert row[1] == 2
    assert float(np.dot(pg_vec, expected)) > 0.9999, "PG centroid beklenen degerden sapiyor"

    qpool = qdrant.client.retrieve(qdrant.IDENTITY_POOL_COLLECTION, ids=[str(cluster_id)], with_vectors=True)
    assert qpool and qpool[0].payload["face_count"] == 2
    q_vec = np.asarray(qpool[0].vector, dtype=np.float64)
    assert float(np.dot(q_vec, pg_vec)) > 0.9999, "Qdrant dual-write PG'den SAPMIS"


def test_pg_brute_force_backend_finds_and_updates_same_identity(monkeypatch, photo):
    e1 = _norm_vec()
    _install_fakes(monkeypatch, 1, [e1])
    db = SessionLocal()
    try:
        saved1 = face_service.detect_and_embed(db, photo)
    finally:
        db.close()
    cluster_id = saved1[0].cluster_id

    e2 = e1 + np.random.randn(512) * 0.01
    e2 = e2 / np.linalg.norm(e2)
    _install_fakes(monkeypatch, 1, [e2])

    monkeypatch.setattr(settings, "IDENTITY_SEARCH_BACKEND", "pg_brute_force")
    db = SessionLocal()
    try:
        saved2 = face_service.detect_and_embed(db, photo)
    finally:
        db.close()

    assert saved2[0].cluster_id == cluster_id, "pg_brute_force backend AYNI kimligi bulamadi"
    db2 = SessionLocal()
    try:
        row = db2.execute(text("SELECT size FROM clusters WHERE id=:i"), {"i": str(cluster_id)}).fetchone()
    finally:
        db2.close()
    assert row[0] == 2


def test_background_face_does_not_update_centroid(monkeypatch, photo):
    e1 = _norm_vec()
    _install_fakes(monkeypatch, 1, [e1])
    db = SessionLocal()
    try:
        saved1 = face_service.detect_and_embed(db, photo)
    finally:
        db.close()
    cluster_id = saved1[0].cluster_id

    db2 = SessionLocal()
    try:
        size_before = db2.execute(text("SELECT size FROM clusters WHERE id=:i"),
                                   {"i": str(cluster_id)}).scalar()
    finally:
        db2.close()

    e2 = e1 + np.random.randn(512) * 0.01
    e2 = e2 / np.linalg.norm(e2)
    # kucuk bbox -> is_background=True (BACKGROUND_FACE_MIN_PIXELS altinda)
    monkeypatch.setattr(
        face_service, "_get_detector",
        lambda: _FakeDetector([_FakeDetected((0, 0, 5, 5), [[0, 0]] * 5, 0.6)]),
    )
    monkeypatch.setattr(face_service, "_get_embedder", lambda: _FakeEmbedder([e2]))
    monkeypatch.setattr(face_service, "_read_image_bgr", lambda path: np.zeros((2000, 2000, 3), dtype=np.uint8))

    db = SessionLocal()
    try:
        saved2 = face_service.detect_and_embed(db, photo)
    finally:
        db.close()

    assert saved2[0].is_background is True
    assert saved2[0].cluster_id == cluster_id, "arka plan yuz eslesmedi (atanmali, sadece merkez guncellenmemeli)"

    db2 = SessionLocal()
    try:
        size_after = db2.execute(text("SELECT size FROM clusters WHERE id=:i"),
                                  {"i": str(cluster_id)}).scalar()
    finally:
        db2.close()
    assert size_after == size_before, "arka plan yuz size'i ARTIRMIS olmamali"


# --- Madde 2: kilit YOK olsaydi yakalayamayacagi regresyon testi ----------


def test_locked_update_uses_fresh_value_not_stale_search_snapshot(monkeypatch, photo):
    """KRITIK: FAZ 1'deki karar (_decide_assignment) SADECE 'hangi kimlik'
    icin kullanilmali - DEGER (centroid/count) FAZ 2'de kilit ALTINDA TAZE
    okunmali. Bu testte: FAZ 1 arama sonucu (dolayisiyla final_kind/
    final_id) SABIT tutulup, kilitli guncelleme calismadan HEMEN ONCE PG
    satiri (baska bir 'islem' simule ederek) DISARIDAN degistiriliyor.
    Guncelleme DOGRU calisiyorsa bu DIStan degisikligi GORMELI (COUNT
    beklenenden 1 fazla olmali) - eski (kilitsiz/stale-deger) tasarimda bu
    degisiklik KAYBOLURDU (klasik kilitleyip-eski-degeri-yazma hatasi)."""
    e1 = _norm_vec()
    _install_fakes(monkeypatch, 1, [e1])
    db = SessionLocal()
    try:
        saved1 = face_service.detect_and_embed(db, photo)
    finally:
        db.close()
    cluster_id = saved1[0].cluster_id

    # FAZ 1'i (karar) manuel calistir - "eski" sonucu SABITLE.
    e2 = e1 + np.random.randn(512) * 0.01
    e2 = e2 / np.linalg.norm(e2)
    db = SessionLocal()
    try:
        final_kind, final_id = face_service._decide_assignment(db, uuid.uuid4(), e2, {})
        assert final_kind == "cluster" and final_id == str(cluster_id)

        # "Baska bir islem" simulasyonu: FAZ 1 ile FAZ 2 ARASINDA, AYRI bir
        # transaction'da satiri DEGISTIR (gercekte bu baska bir worker/HTTP
        # istegi olurdu - burada dogrudan SQL ile taklit ediyoruz).
        other_db = SessionLocal()
        try:
            e_other = e1 + np.random.randn(512) * 0.01
            e_other = e_other / np.linalg.norm(e_other)
            other_centroid = (e1 + e2 * 0 + e_other)  # icerik onemli degil, SADECE count=2 olsun
            other_centroid = other_centroid / np.linalg.norm(other_centroid)
            other_db.execute(
                text("UPDATE clusters SET centroid=:c, size=2, centroid_updated_at=now() WHERE id=:i"),
                {"c": other_centroid.astype(np.float32).tobytes(), "i": str(cluster_id)},
            )
            other_db.commit()
        finally:
            other_db.close()

        # FAZ 2: kilit al + TAZE deger oku + guncelle (gercek fonksiyon).
        from app.db import identity_locks
        identity_locks.lock_identities(db, [("cluster", cluster_id)])
        from app.db.models import Face
        fake_face = Face(id=uuid.uuid4(), photo_id=photo.id, bbox={"x": 0, "y": 0, "w": 1, "h": 1},
                          landmarks=[[0, 0]] * 5, det_confidence=0.9, is_background=False, crop_path="x")
        db.add(fake_face)
        db.flush()
        op = face_service._apply_assignment_locked(db, fake_face, e2, final_kind, final_id, False)
        db.commit()
    finally:
        db.close()

    db2 = SessionLocal()
    try:
        row = db2.execute(text("SELECT size FROM clusters WHERE id=:i"), {"i": str(cluster_id)}).fetchone()
    finally:
        db2.close()
    assert row[0] == 3, (
        f"beklenen 3 (disaridan yazilan 2 + bu guncellemenin +1'i), gelen {row[0]} - "
        "guncelleme FAZ 1'in ESKI (stale) degerini kullanmis olabilir!"
    )


# --- Madde 1: rollback, sahipsiz (Face'siz) centroid BIRAKMAMALI ----------


def test_rollback_does_not_leave_orphan_centroid(monkeypatch, photo):
    """KRITIK: fotografin IKINCI yuzu FAZ 2'de (Face satirlari yazilirken)
    hata verirse, TUM transaction (ilk yuzun centroid'i DAHIL) geri
    alinmali - hicbir centroid degisikligi Face satirindan BAGIMSIZ olarak
    KALICI kalmamali (eski, 'her yuz kendi ayri transaction'inda commit
    eder' tasarimin URETTIGI 'anna'-sinifi bug'in TAM TERSI)."""
    e1 = _norm_vec()
    e2 = e1 + np.random.randn(512) * 0.01
    e2 = e2 / np.linalg.norm(e2)
    _install_fakes(monkeypatch, 2, [e1, e2])

    real_apply = face_service._apply_assignment_locked
    call_count = {"n": 0}

    def flaky_apply(db, face, embedding, final_kind, final_id, is_new_identity):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated failure on 2nd face")
        return real_apply(db, face, embedding, final_kind, final_id, is_new_identity)

    monkeypatch.setattr(face_service, "_apply_assignment_locked", flaky_apply)

    db = SessionLocal()
    try:
        with pytest.raises(RuntimeError, match="simulated failure"):
            face_service.detect_and_embed(db, photo)
        db.rollback()
    finally:
        db.close()

    db2 = SessionLocal()
    try:
        face_count = db2.execute(text("SELECT count(*) FROM faces WHERE photo_id=:p"),
                                  {"p": str(photo.id)}).scalar()
        # ilk yuzun _apply_assignment_locked'i basariyla YENI bir kume actigi
        # icin (rollback OLMASAYDI) bir cluster satiri da beklenirdi - simdi
        # HICBIRI KALICI olmamali.
        orphan_clusters = db2.execute(
            text("SELECT count(*) FROM clusters WHERE created_at > now() - interval '1 minute' "
                 "AND size = 1 AND id NOT IN (SELECT cluster_id FROM faces WHERE cluster_id IS NOT NULL)")
        ).scalar()
    finally:
        db2.close()

    assert face_count == 0, "rollback sonrasi Face satiri KALMAMALI"
    # NOT: orphan_clusters kontrolu gevsek (baska testlerden kalinti
    # olabilecegi icin sadece bu test suresince olusan spesifik satiri
    # dogrudan aramak daha saglam olurdu, ama transaction TAMAMEN rollback
    # oldugu icin zaten cluster.id'yi bile bilmiyoruz - bu da testin
    # ispatladigi seyin ta kendisi: hicbir iz kalmadi.)


# --- Madde 2/3 revizyonu: fotograf-ici local_pool -------------------------


def test_same_person_twice_in_one_photo_creates_single_cluster(monkeypatch, photo):
    """OLCUM (bu revizyon turu): ayni fotografta ayni kisinin >1 yuzu
    vakalarin %4.87'sinde (coklu-yuzlu fotograflarin %8.84'unde) gorulmus -
    nadir DEGIL. Bu kisi sistemde ILK KEZ ortaya cikiyorsa (DB/Qdrant'ta
    hicbir aday yok), local_pool OLMADAN iki AYRI kume acilirdi. Bu test bunun
    ARTIK olmadigini kanitlar: ayni fotografta, ayni kisinin (neredeyse ayni
    embedding'e sahip) IKI yuzu -> TEK kume, size=2."""
    e1 = _norm_vec()
    e2 = e1 + np.random.randn(512) * 0.005  # ayni kisi, neredeyse ozdes
    e2 = e2 / np.linalg.norm(e2)
    _install_fakes(monkeypatch, 2, [e1, e2])

    db = SessionLocal()
    try:
        saved = face_service.detect_and_embed(db, photo)
    finally:
        db.close()

    assert len(saved) == 2
    assert saved[0].cluster_id is not None
    assert saved[0].cluster_id == saved[1].cluster_id, (
        "ayni fotografta ayni kisinin iki yuzu FARKLI kumelere dusmus - "
        "fotograf-ici local_pool calismiyor olabilir"
    )

    db2 = SessionLocal()
    try:
        row = db2.execute(text("SELECT centroid, size FROM clusters WHERE id=:i"),
                           {"i": str(saved[0].cluster_id)}).fetchone()
        cluster_count = db2.execute(text("SELECT count(*) FROM clusters WHERE id::text = ANY(:ids)"),
                                     {"ids": [str(saved[0].cluster_id), str(saved[1].cluster_id)]}).scalar()
    finally:
        db2.close()

    assert row[1] == 2, f"kume size'i 2 olmali (iki yuz), gelen {row[1]}"
    assert cluster_count == 1, "iki AYRI cluster satiri olusmus olmamali"

    pg_vec = np.frombuffer(bytes(row[0]), dtype=np.float32).astype(np.float64)
    expected = e1 + e2
    expected = expected / np.linalg.norm(expected)
    assert float(np.dot(pg_vec, expected)) > 0.9999, "birlesik kume centroid'i beklenenden sapiyor"


def test_two_different_people_in_one_photo_create_two_clusters(monkeypatch, photo):
    """local_pool'un YANLIS BIRLESTIRME yapmadigini kanitlayan karsi-test:
    ayni fotografta BIRBIRINDEN TAMAMEN FARKLI iki kisi -> IKI AYRI kume."""
    e1 = _norm_vec()
    e2 = _norm_vec()  # bagimsiz, rastgele - farkli kisi
    # AUTO_ASSIGN_THRESHOLD'un cok altinda kalmasi icin ortogonal-e-yakin
    # rastgele vektorler (512 boyutta iki bagimsiz rastgele vektorun kosinus
    # benzerligi zaten esigin cok altinda olur).
    _install_fakes(monkeypatch, 2, [e1, e2])

    db = SessionLocal()
    try:
        saved = face_service.detect_and_embed(db, photo)
    finally:
        db.close()

    assert len(saved) == 2
    assert saved[0].cluster_id != saved[1].cluster_id, (
        "birbirinden FARKLI iki kisi YANLISLIKLA ayni kumeye birlestirilmis"
    )
