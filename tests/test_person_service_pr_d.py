"""PR-D testleri: label_cluster/merge_identities/reassign_face/delete_photo'nun
PG centroid'i (persons.centroid/clusters.centroid) Qdrant identity_pool ile
tutarli guncelledigini, deadlock yaratmadigini ve Katman 2'nin (_upsert_
identity_centroid) Katman 1'in ciktisindan TAMAMEN BAGIMSIZ (taze sorgulu)
oldugunu dogrular. GERCEK PostgreSQL + Qdrant'a karsi calisir.
"""

import threading
import uuid
from datetime import datetime

import numpy as np
import pytest
from qdrant_client.models import PointStruct
from sqlalchemy import text

from app.db import identity_locks, qdrant
from app.db.models import Cluster, Face, Person, Photo
from app.db.session import SessionLocal
from app.services import face_service, person_service, photo_service


def _norm_vec():
    v = np.random.randn(512)
    return v / np.linalg.norm(v)


def _mk_photo(db, user_id):
    p = Photo(filename="t.jpg", storage_path=f"uploads/{uuid.uuid4()}.jpg", uploaded_by_user_id=user_id)
    db.add(p)
    db.flush()
    return p


def _mk_face(db, photo_id, embedding, person_id=None, cluster_id=None):
    f = Face(
        photo_id=photo_id, bbox={"x": 0, "y": 0, "w": 100, "h": 100}, landmarks=[[0, 0]] * 5,
        det_confidence=0.95, quality_score=0.9, is_background=False, crop_path="x",
        person_id=person_id, cluster_id=cluster_id,
    )
    db.add(f)
    db.flush()
    qdrant.client.upsert(qdrant.FACES_COLLECTION, points=[
        PointStruct(id=str(f.id), vector=embedding.tolist(), payload={})
    ])
    return f


def _pg_cosine_vs_qdrant(table, identity_id):
    db = SessionLocal()
    try:
        row = db.execute(text(f"SELECT centroid FROM {table} WHERE id=:i"), {"i": str(identity_id)}).fetchone()
    finally:
        db.close()
    assert row is not None and row[0] is not None, f"{table}.centroid NULL/yok: {identity_id}"
    pg_vec = np.frombuffer(bytes(row[0]), dtype=np.float32).astype(np.float64)

    q = qdrant.client.retrieve(qdrant.IDENTITY_POOL_COLLECTION, ids=[str(identity_id)], with_vectors=True)
    assert q, f"identity_pool'da nokta yok: {identity_id}"
    q_vec = np.asarray(q[0].vector, dtype=np.float64)

    return float(np.dot(pg_vec, q_vec) / (np.linalg.norm(pg_vec) * np.linalg.norm(q_vec)))


@pytest.fixture
def cleanup_registry():
    """Testin olusturdugu (kind,id)/photo/user kayitlarini toplar, sonda temizler."""
    registry = {"persons": [], "clusters": [], "photos": [], "faces": []}
    yield registry
    db = SessionLocal()
    try:
        face_ids = [str(i) for i in registry["faces"]]
        if face_ids:
            qdrant.client.delete(qdrant.FACES_COLLECTION, points_selector=face_ids)
            # merge_identities/reassign_face bu yuzlere must_link/cannot_link
            # kisiti birakiyor - FK'yi once temizle.
            db.execute(
                text(
                    "DELETE FROM cluster_constraints WHERE face_id_a::text = ANY(:ids) "
                    "OR face_id_b::text = ANY(:ids)"
                ),
                {"ids": face_ids},
            )
            db.execute(text("DELETE FROM faces WHERE id::text = ANY(:ids)"), {"ids": face_ids})
        pool_ids = [str(i) for i in registry["persons"] + registry["clusters"]]
        if pool_ids:
            qdrant.client.delete(qdrant.IDENTITY_POOL_COLLECTION, points_selector=pool_ids)
        if registry["persons"]:
            db.execute(text("DELETE FROM persons WHERE id::text = ANY(:ids)"),
                       {"ids": [str(i) for i in registry["persons"]]})
        if registry["clusters"]:
            db.execute(text("DELETE FROM clusters WHERE id::text = ANY(:ids)"),
                       {"ids": [str(i) for i in registry["clusters"]]})
        if registry["photos"]:
            db.execute(text("DELETE FROM photos WHERE id::text = ANY(:ids)"),
                       {"ids": [str(i) for i in registry["photos"]]})
        db.commit()
    finally:
        db.close()


# --- Madde 6: her fonksiyon icin PG<->Qdrant kosinus benzerligi ----------


def test_label_cluster_pg_matches_qdrant(test_user_id, cleanup_registry):
    db = SessionLocal()
    try:
        photo = _mk_photo(db, test_user_id)
        cluster = Cluster(status="unlabeled", size=0, created_at=datetime.utcnow())
        db.add(cluster)
        db.flush()
        e1, e2 = _norm_vec(), _norm_vec()
        f1 = _mk_face(db, photo.id, e1, cluster_id=cluster.id)
        f2 = _mk_face(db, photo.id, e2, cluster_id=cluster.id)
        cluster.size = 2
        db.commit()
        cleanup_registry["photos"].append(photo.id)
        cleanup_registry["clusters"].append(cluster.id)
        cleanup_registry["faces"] += [f1.id, f2.id]

        person = person_service.label_cluster(db, cluster.id, "Test", test_user_id)
        cleanup_registry["persons"].append(person.id)
    finally:
        db.close()

    sim = _pg_cosine_vs_qdrant("persons", person.id)
    assert sim > 0.9999, f"label_cluster sonrasi PG/Qdrant kosinus benzerligi dusuk: {sim}"


def test_merge_identities_pg_matches_qdrant(test_user_id, cleanup_registry):
    db = SessionLocal()
    try:
        photo = _mk_photo(db, test_user_id)
        target = Person(display_name="T", face_count=1, created_by_user_id=test_user_id, created_at=datetime.utcnow())
        db.add(target)
        db.flush()
        e1 = _norm_vec()
        f1 = _mk_face(db, photo.id, e1, person_id=target.id)

        source_cluster = Cluster(status="unlabeled", size=1, created_at=datetime.utcnow())
        db.add(source_cluster)
        db.flush()
        e2 = _norm_vec()
        f2 = _mk_face(db, photo.id, e2, cluster_id=source_cluster.id)
        db.commit()

        cleanup_registry["photos"].append(photo.id)
        cleanup_registry["persons"].append(target.id)
        cleanup_registry["clusters"].append(source_cluster.id)
        cleanup_registry["faces"] += [f1.id, f2.id]

        target_id = target.id
        person_service.merge_identities(db, "person", target_id, "cluster", source_cluster.id, test_user_id)
    finally:
        db.close()

    sim = _pg_cosine_vs_qdrant("persons", target_id)
    assert sim > 0.9999, f"merge_identities sonrasi PG/Qdrant kosinus benzerligi dusuk: {sim}"


def test_merge_identities_zeroes_source_face_count(test_user_id, cleanup_registry):
    """Madde 3 ("anna" vakasi) regresyon testi."""
    db = SessionLocal()
    try:
        photo = _mk_photo(db, test_user_id)
        target = Person(display_name="T", face_count=1, created_by_user_id=test_user_id, created_at=datetime.utcnow())
        source = Person(display_name="S", face_count=2, created_by_user_id=test_user_id, created_at=datetime.utcnow())
        db.add_all([target, source])
        db.flush()
        e1, e2 = _norm_vec(), _norm_vec()
        f1 = _mk_face(db, photo.id, e1, person_id=target.id)
        f2 = _mk_face(db, photo.id, e2, person_id=source.id)
        db.commit()

        cleanup_registry["photos"].append(photo.id)
        cleanup_registry["persons"] += [target.id, source.id]
        cleanup_registry["faces"] += [f1.id, f2.id]

        target_id, source_id = target.id, source.id
        person_service.merge_identities(db, "person", target_id, "person", source_id, test_user_id)
    finally:
        db.close()

    db2 = SessionLocal()
    try:
        row = db2.execute(text("SELECT face_count, deleted_at FROM persons WHERE id=:i"),
                           {"i": str(source_id)}).fetchone()
    finally:
        db2.close()
    assert row.deleted_at is not None, "source soft-delete edilmemis"
    assert row.face_count == 0, f"source.face_count sifirlanmamis (eski 'anna' bug'i): {row.face_count}"


def test_reassign_face_pg_matches_qdrant_for_both_identities(test_user_id, cleanup_registry):
    db = SessionLocal()
    try:
        photo = _mk_photo(db, test_user_id)
        person_a = Person(display_name="A", face_count=2, created_by_user_id=test_user_id, created_at=datetime.utcnow())
        person_b = Person(display_name="B", face_count=1, created_by_user_id=test_user_id, created_at=datetime.utcnow())
        db.add_all([person_a, person_b])
        db.flush()
        e1, e2, e3 = _norm_vec(), _norm_vec(), _norm_vec()
        f1 = _mk_face(db, photo.id, e1, person_id=person_a.id)
        f2 = _mk_face(db, photo.id, e2, person_id=person_a.id)  # bu tasinacak
        f3 = _mk_face(db, photo.id, e3, person_id=person_b.id)
        db.commit()

        cleanup_registry["photos"].append(photo.id)
        cleanup_registry["persons"] += [person_a.id, person_b.id]
        cleanup_registry["faces"] += [f1.id, f2.id, f3.id]

        person_a_id, person_b_id, f2_id = person_a.id, person_b.id, f2.id
        person_service.reassign_face(db, f2_id, person_b_id, test_user_id)
    finally:
        db.close()

    sim_a = _pg_cosine_vs_qdrant("persons", person_a_id)
    sim_b = _pg_cosine_vs_qdrant("persons", person_b_id)
    assert sim_a > 0.9999, f"reassign sonrasi ESKI kimlik (A) PG/Qdrant sapmis: {sim_a}"
    assert sim_b > 0.9999, f"reassign sonrasi YENI kimlik (B) PG/Qdrant sapmis: {sim_b}"


def test_delete_photo_pg_matches_qdrant_for_remaining_person(test_user_id, cleanup_registry):
    db = SessionLocal()
    try:
        photo1 = _mk_photo(db, test_user_id)
        photo2 = _mk_photo(db, test_user_id)
        person = Person(display_name="P", face_count=2, created_by_user_id=test_user_id, created_at=datetime.utcnow())
        db.add(person)
        db.flush()
        e1, e2 = _norm_vec(), _norm_vec()
        f1 = _mk_face(db, photo1.id, e1, person_id=person.id)
        f2 = _mk_face(db, photo2.id, e2, person_id=person.id)
        db.commit()

        cleanup_registry["photos"].append(photo2.id)
        cleanup_registry["persons"].append(person.id)
        cleanup_registry["faces"].append(f2.id)

        person_id, photo1_id = person.id, photo1.id
        photo_service.delete_photo(db, photo1_id)
    finally:
        db.close()

    sim = _pg_cosine_vs_qdrant("persons", person_id)
    assert sim > 0.9999, f"delete_photo sonrasi kalan person PG/Qdrant sapmis: {sim}"


# --- Madde 6: deadlock - merge_identities ters sirada, iki GERCEK thread --


def test_merge_identities_opposite_order_does_not_deadlock(test_user_id, cleanup_registry):
    db = SessionLocal()
    try:
        photo = _mk_photo(db, test_user_id)
        p1 = Person(display_name="P1", face_count=1, created_by_user_id=test_user_id, created_at=datetime.utcnow())
        p2 = Person(display_name="P2", face_count=1, created_by_user_id=test_user_id, created_at=datetime.utcnow())
        db.add_all([p1, p2])
        db.flush()
        e1, e2 = _norm_vec(), _norm_vec()
        f1 = _mk_face(db, photo.id, e1, person_id=p1.id)
        f2 = _mk_face(db, photo.id, e2, person_id=p2.id)
        db.commit()

        cleanup_registry["photos"].append(photo.id)
        cleanup_registry["persons"] += [p1.id, p2.id]
        cleanup_registry["faces"] += [f1.id, f2.id]
        p1_id, p2_id = p1.id, p2.id
    finally:
        db.close()

    results = {}
    barrier = threading.Barrier(2, timeout=10)

    def worker(name, target_id, source_id):
        db = SessionLocal()
        try:
            barrier.wait()
            person_service.merge_identities(db, "person", target_id, "person", source_id, test_user_id)
            results[name] = "ok"
        except ValueError as exc:
            results[name] = f"value_error: {exc}"
        except Exception as exc:  # pragma: no cover
            results[name] = f"error: {exc}"
        finally:
            db.close()

    t1 = threading.Thread(target=worker, args=("t1", p1_id, p2_id))  # target=P1, source=P2
    t2 = threading.Thread(target=worker, args=("t2", p2_id, p1_id))  # target=P2, source=P1 (TERS)
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)

    assert not t1.is_alive() and not t2.is_alive(), f"DEADLOCK - thread(ler) 15sn'de bitmedi: {results}"
    # Iki merge AYNI ciftle cakisiyor - biri GERCEKTEN birlesir, digeri
    # (kaynagi bosalmis bulur) is-kurali hatasi alir. Deadlock OLMAMASI
    # onemli olan; hangisinin "kazandigi" onemli degil.
    assert set(results.values()) <= {
        "ok",
        "value_error: Klasor(ler) artik mevcut degil ya da bos - muhtemelen daha once "
        "birlestirilmis/silinmis. Oneri listesini yenileyip tekrar deneyin.",
    }, results
    assert "ok" in results.values(), f"beklenmedik: hicbir merge basarili olmadi: {results}"


# --- Madde 6 (kullanicinin duzeltmesi): Katman 2, Katman 1'den BAGIMSIZ --


def test_layer2_sees_faces_added_between_layer1_commit_and_layer2_start(test_user_id, cleanup_registry):
    """KRITIK: _upsert_identity_centroid (Katman 2) face_ids PARAMETRE
    OLARAK ALMAZ - kendi kilidini aldiktan SONRA Face tablosundan TAZE
    sorgular. Bu test: Katman 1 commit eder, ARADA (Katman 2 HENUZ
    baslamadan) 'worker tarzi' bir surec AYNI kimlige (kendi kilidiyle) yeni
    bir yuz ekleyip commit eder - Katman 2'nin urettigi centroid'in BU YENI
    yuzu de icermesi gerekir (eski, face_ids-parametreli tasarimda bu
    KAYBOLURDU)."""
    db = SessionLocal()
    try:
        photo = _mk_photo(db, test_user_id)
        person = Person(display_name="T", face_count=1, created_by_user_id=test_user_id, created_at=datetime.utcnow())
        db.add(person)
        db.flush()
        e1 = _norm_vec()
        f1 = _mk_face(db, photo.id, e1, person_id=person.id)
        db.commit()
        person_id = person.id

        cleanup_registry["photos"].append(photo.id)
        cleanup_registry["persons"].append(person_id)
        cleanup_registry["faces"].append(f1.id)
    finally:
        db.close()

    # --- "Katman 1": erken kilit alip commit eder (baska bir is yapmadan -
    # sadece kilit-al-commit-et deseninin kendisini temsil eder) ---
    layer1_db = SessionLocal()
    try:
        identity_locks.lock_identities(layer1_db, [("person", person_id)])
        layer1_db.commit()
    finally:
        layer1_db.close()

    # --- "worker tarzi yeni yuz ekleme" - Katman 1 SONRASI, Katman 2 ONCESI ---
    e2 = _norm_vec()
    worker_db = SessionLocal()
    try:
        identity_locks.lock_identities(worker_db, [("person", person_id)])
        f2 = Face(
            photo_id=photo.id, bbox={"x": 0, "y": 0, "w": 100, "h": 100}, landmarks=[[0, 0]] * 5,
            det_confidence=0.95, quality_score=0.9, is_background=False, crop_path="x", person_id=person_id,
        )
        worker_db.add(f2)
        worker_db.flush()
        qdrant.client.upsert(qdrant.FACES_COLLECTION, points=[PointStruct(id=str(f2.id), vector=e2.tolist(), payload={})])
        worker_db.query(Person).filter(Person.id == person_id).update({"face_count": 2})
        worker_db.commit()
        cleanup_registry["faces"].append(f2.id)
    finally:
        worker_db.close()

    # --- Katman 2 ---
    db2 = SessionLocal()
    try:
        op = person_service._upsert_identity_centroid(db2, "person", person_id)
    finally:
        db2.close()

    assert op is not None
    assert op["payload"]["face_count"] == 2, (
        f"Katman 2, ARADA eklenen yuzu KACIRMIS - face_count={op['payload']['face_count']} (2 olmali). "
        "face_ids parametre olarak gelseydi bu test BASARISIZ olurdu."
    )
    expected = e1 + e2
    expected = expected / np.linalg.norm(expected)
    result_vec = np.array(op["centroid"])
    assert float(np.dot(result_vec, expected)) > 0.9999, "Katman 2'nin urettigi centroid beklenenden sapiyor"


# --- Madde 6/9 bulgusunun duzeltmesi: liveness filtresi -------------------


def test_upsert_identity_centroid_liveness_filter_does_not_block_active_identity(test_user_id, cleanup_registry):
    """Madde 3: liveness filtresi (deleted_at IS NULL / status='unlabeled')
    AKTIF bir kimligin KENDI merkez guncellemesini ENGELLEMEMELI - bu, tam
    olarak merge_identities'in TARGET icin yaptigi cagriyla AYNI durum
    (target hicbir zaman silinmez/retire edilmez, sadece uyeligi degisir)."""
    db = SessionLocal()
    try:
        photo = _mk_photo(db, test_user_id)
        person = Person(display_name="Active", face_count=1, created_by_user_id=test_user_id,
                         created_at=datetime.utcnow())
        db.add(person)
        db.flush()
        e1 = _norm_vec()
        f1 = _mk_face(db, photo.id, e1, person_id=person.id)
        db.commit()
        person_id = person.id

        cleanup_registry["photos"].append(photo.id)
        cleanup_registry["persons"].append(person_id)
        cleanup_registry["faces"].append(f1.id)

        op = person_service._upsert_identity_centroid(db, "person", person_id)
    finally:
        db.close()

    assert op is not None, (
        "aktif (deleted_at IS NULL) bir kimlik icin liveness filtresi YANLISLIKLA "
        "engellemis - merge_identities'in target guncellemesi de bu yolu kullaniyor"
    )
    assert op["payload"]["face_count"] == 1


def test_stale_decision_prevents_ghost_resurrection_after_merge(test_user_id, cleanup_registry):
    """Madde 6/9 bulgusunun asil regresyon testi - GERCEK 2 thread/2 session:

    Senaryo: worker'in FAZ 1'i (bu testte, "Mehmet" kimligine karar vermis
    bir onceki arama olarak TEMSIL EDILIYOR) Mehmet'i bulmus. TAM O SIRADA
    admin Mehmet'i Ahmet'e merge ediyor (GERCEK bir thread'de, GERCEK bir
    session'da, merge_identities). Worker'in FAZ 2'si (identity_locks.
    lock_identities + _apply_assignment_locked) merge'un TAMAMEN BITMESINI
    bekler (Event ile GARANTI edilir - Postgres satir kilidi zaten bunu
    saglar, Event sadece testi non-flaky yapmak icin merge'un BASLAMASINI
    degil BITMESINI garanti eder), sonra Mehmet'in ARTIK GECERSIZ oldugunu
    (StaleIdentityDecision) fark etmeli - Mehmet'in face_count'u ARTMAMALI,
    Qdrant identity_pool noktasi GERI GELMEMELI ("hayalet kayit" olusmamali).
    """
    db = SessionLocal()
    try:
        photo = _mk_photo(db, test_user_id)
        ahmet = Person(display_name="Ahmet", face_count=1, created_by_user_id=test_user_id,
                        created_at=datetime.utcnow())
        mehmet = Person(display_name="Mehmet", face_count=1, created_by_user_id=test_user_id,
                         created_at=datetime.utcnow())
        db.add_all([ahmet, mehmet])
        db.flush()
        e_ahmet, e_mehmet = _norm_vec(), _norm_vec()
        f_ahmet = _mk_face(db, photo.id, e_ahmet, person_id=ahmet.id)
        f_mehmet = _mk_face(db, photo.id, e_mehmet, person_id=mehmet.id)
        db.commit()
        ahmet_id, mehmet_id, photo_id = ahmet.id, mehmet.id, photo.id

        cleanup_registry["photos"].append(photo.id)
        cleanup_registry["persons"] += [ahmet_id, mehmet_id]
        cleanup_registry["faces"] += [f_ahmet.id, f_mehmet.id]
    finally:
        db.close()

    merge_done = threading.Event()
    worker_result = {}
    new_face_ids = []

    def admin_merge():
        db = SessionLocal()
        try:
            person_service.merge_identities(db, "person", ahmet_id, "person", mehmet_id, test_user_id)
        finally:
            db.close()
        merge_done.set()

    def worker_faz2():
        # FAZ 1'in (bu turdan ONCE verilmis, dolayisiyla BAYATLAYACAK)
        # karari: Mehmet. merge_done.wait() GERCEK bir thread'in GERCEK bir
        # merge_identities cagrisinin TAMAMEN bitmesini bekliyor - bu,
        # "worker Faz 2'ye merge SONRASI giriyor" senaryosunu deterministik
        # kilar (Postgres'in kendi satir kilidi zaten ayni sirayi
        # zorlardi, Event flaky olmamasi icin).
        merge_done.wait(timeout=10)
        db = SessionLocal()
        try:
            new_embedding = e_mehmet + np.random.randn(512) * 0.01
            new_embedding = new_embedding / np.linalg.norm(new_embedding)
            new_face = Face(
                photo_id=photo_id, bbox={"x": 0, "y": 0, "w": 100, "h": 100}, landmarks=[[0, 0]] * 5,
                det_confidence=0.95, quality_score=0.9, is_background=False, crop_path="x",
            )
            db.add(new_face)
            db.flush()
            new_face_ids.append(new_face.id)

            identity_locks.lock_identities(db, [("person", mehmet_id)])
            try:
                face_service._apply_assignment_locked(
                    db, new_face, new_embedding, "person", str(mehmet_id), False,
                )
                db.commit()
                worker_result["outcome"] = "wrote"
            except identity_locks.StaleIdentityDecision as exc:
                db.rollback()
                worker_result["outcome"] = "stale_decision"
                worker_result["exc"] = str(exc)
        finally:
            db.close()

    t_worker = threading.Thread(target=worker_faz2)
    t_admin = threading.Thread(target=admin_merge)
    t_worker.start()
    t_admin.start()
    t_worker.join(timeout=15)
    t_admin.join(timeout=15)

    cleanup_registry["faces"] += new_face_ids

    assert worker_result.get("outcome") == "stale_decision", (
        f"Mehmet DIRILMIS OLABILIR - beklenen 'stale_decision', gelen: {worker_result}"
    )

    db2 = SessionLocal()
    try:
        row = db2.execute(
            text("SELECT face_count, deleted_at FROM persons WHERE id=:i"), {"i": str(mehmet_id)}
        ).fetchone()
    finally:
        db2.close()
    assert row.deleted_at is not None, "Mehmet hala soft-delete edilmis olmali"
    assert row.face_count == 0, f"Mehmet'in face_count'u ARTMAMALI (dirilme yok): {row.face_count}"

    q = qdrant.client.retrieve(qdrant.IDENTITY_POOL_COLLECTION, ids=[str(mehmet_id)])
    assert not q, "Mehmet'in Qdrant identity_pool noktasi GERI GELMEMELI (hayalet kayit)"


# --- Bayat oneri duzeltmesi (Seviye 1): reddedilmis cift birlestirilemez --


def test_merge_identities_rejects_previously_rejected_pair(test_user_id, cleanup_registry):
    """reject_merge ile 'ayni kisi DEGIL' denmis bir cift, dogrudan bir
    merge_identities cagrisiyla (bayat bir oneriyi kabul etmek dahil) YINE DE
    birlestirilememeli."""
    db = SessionLocal()
    try:
        photo = _mk_photo(db, test_user_id)
        p1 = Person(display_name="P1", face_count=1, created_by_user_id=test_user_id, created_at=datetime.utcnow())
        p2 = Person(display_name="P2", face_count=1, created_by_user_id=test_user_id, created_at=datetime.utcnow())
        db.add_all([p1, p2])
        db.flush()
        e1, e2 = _norm_vec(), _norm_vec()
        f1 = _mk_face(db, photo.id, e1, person_id=p1.id)
        f2 = _mk_face(db, photo.id, e2, person_id=p2.id)
        db.commit()

        cleanup_registry["photos"].append(photo.id)
        cleanup_registry["persons"] += [p1.id, p2.id]
        cleanup_registry["faces"] += [f1.id, f2.id]
        p1_id, p2_id = p1.id, p2.id

        person_service.reject_merge(
            db, [{"kind": "person", "id": p1_id}, {"kind": "person", "id": p2_id}], test_user_id
        )

        with pytest.raises(ValueError, match="ayni kisi degil"):
            person_service.merge_identities(db, "person", p1_id, "person", p2_id, test_user_id)

        # Ters yonde de (source/target yer degistirse bile) engellenmeli.
        with pytest.raises(ValueError, match="ayni kisi degil"):
            person_service.merge_identities(db, "person", p2_id, "person", p1_id, test_user_id)
    finally:
        db.close()

    # Hicbir merge gerceklesmemis olmali - ikisi de hala aktif, ayri kisiler.
    db2 = SessionLocal()
    try:
        rows = db2.execute(
            text("SELECT id, deleted_at FROM persons WHERE id::text = ANY(:ids)"),
            {"ids": [str(p1_id), str(p2_id)]},
        ).fetchall()
    finally:
        db2.close()
    assert len(rows) == 2
    assert all(r.deleted_at is None for r in rows), "engellenen merge YINE DE gerceklesmis"


def test_merge_identities_unblocked_pair_still_works(test_user_id, cleanup_registry):
    """Karsi-test: aralarinda cannot_link OLMAYAN normal bir cift, yeni
    kontrol yuzunden YANLISLIKLA engellenmemeli."""
    db = SessionLocal()
    try:
        photo = _mk_photo(db, test_user_id)
        target = Person(display_name="T", face_count=1, created_by_user_id=test_user_id, created_at=datetime.utcnow())
        source = Person(display_name="S", face_count=1, created_by_user_id=test_user_id, created_at=datetime.utcnow())
        db.add_all([target, source])
        db.flush()
        e1, e2 = _norm_vec(), _norm_vec()
        f1 = _mk_face(db, photo.id, e1, person_id=target.id)
        f2 = _mk_face(db, photo.id, e2, person_id=source.id)
        db.commit()

        cleanup_registry["photos"].append(photo.id)
        cleanup_registry["persons"] += [target.id, source.id]
        cleanup_registry["faces"] += [f1.id, f2.id]

        target_id, source_id = target.id, source.id
        result = person_service.merge_identities(db, "person", target_id, "person", source_id, test_user_id)
    finally:
        db.close()

    assert result["id"] == str(target_id)
