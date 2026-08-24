"""HDBSCAN oneri motoru - suggest_merges(), Bolum 8.4.

Kumeleme mantigi (cluster_embeddings) clustering_reference.py'den birebir
alinmistir - Colab'da 30 fotograf/5 kisilik POC setinde dogrulandi (0 gurultu,
5 kume, %100 saflik). suggest_merges(), zaten kimlik atanmis yuzleri (person_id
ya da cluster_id) HDBSCAN ile YENIDEN gruplayip, gercek-zamanli atama
katmaninin (face_service._assign_or_bucket) sira-bagimli dogasi yuzunden
olusan hatali bolunmeleri geriye donuk tespit eder - salt okunur bir tarama,
hicbir sey degistirmez/silmez/tasimaz (bkz. asagidaki fonksiyon docstring'i).

NOT (temizlik gecmisi): bu modulde daha once, tum isimsiz klasorleri silip
HDBSCAN ile SIFIRDAN kuran YIKICI bir "gece isi" (run_nightly_clustering)
vardi - hicbir uc noktaya hic baglanmadi, kullanicinin acik istegiyle
kaldirildi (kod tabaninda olu agirlik olarak durmasin diye).
"""

import uuid
from collections import defaultdict

import hdbscan
import numpy as np
from sqlalchemy.orm import Session

from app.db import qdrant
from app.db.models import Cluster, ClusterConstraint, Face, Person

# Dokumandaki baslangic parametreleri (Bolum 8.3) - POC ile kalibre edilecek.
MIN_CLUSTER_SIZE = 3
MIN_SAMPLES = 2


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    min_samples: int = MIN_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dokumandaki baslangic parametreleriyle (Bolum 8.3) HDBSCAN kumeleme yapar.

    embeddings: (N, 512) sekilli, her satiri L2-normalize edilmis embedding
        matrisi (FaceEmbedder.get_embedding() ciktilari).

    Onemli: `hdbscan` kutuphanesi metric='cosine'i dogrudan desteklemiyor
    (ball-tree backend'i kabul etmiyor). Embedding'ler zaten normalize
    (norm=1) oldugu icin metric='euclidean' matematiksel olarak cosine ile
    ESDEGER siralama/kumeleme sonucu verir - dokumandaki "metric=cosine"
    kararindan sapma degil, ayni seyin pratik uygulamasi.

    Donus: (labels, probabilities)
        labels[i] = i. yuzun kume ID'si (-1 = "kimliksiz yuzler havuzu",
            Bolum 8.3 - Gurultu davranisi).
        probabilities[i] = 0-1 arasi uyelik olasiligi.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings.astype(np.float64))
    return labels, clusterer.probabilities_


def suggest_merges(db: Session) -> list[dict]:
    """HDBSCAN katmani (Bolum 8.3 "gece isi"nin YIKICI OLMAYAN hali).

    Ne yapar: tum havuzu (arka plan yuzleri haric) HDBSCAN ile kumeler, sonra
    "ayni HDBSCAN kumesine dustugu halde FARKLI klasorlerde duran" yuzleri
    arar. Boyle bir durum, gercek-zamanli algoritmanin sıraya baglilik yuzunden
    ayni kisiyi bolmus olabilecegini gosterir -> birlestirme onerisi uretir.

    Ne YAPMAZ: hicbir sey silmez, tasimaz, yeniden kurmaz. Yalnizca oneri
    dondurur; uygulama karari insana aittir (B4, Bolum 10.2).

    Tasarim gerekceleri:
      - Arka plan yuzleri disarida: guvenilmez embeddingler (112x112 alti)
        kumelemeyi kirletir (bkz. face_service.BACKGROUND_FACE_*).
      - Gurultuye (-1) dusen yuzlere DOKUNULMAZ: mevcut klasorlerinde kalirlar,
        yani hicbir yuz gorunmez olmaz. Klasik gece isinin en buyuk riski buydu.
      - cannot_link kisitlari saygi gorur: kullanici "bunlar ayni kisi degil"
        demisse o cift tekrar onerilmez.
    """
    faces = (
        db.query(Face)
        .filter(Face.is_background.is_(False))
        .filter((Face.cluster_id.isnot(None)) | (Face.person_id.isnot(None)))
        .all()
    )
    if len(faces) < MIN_CLUSTER_SIZE:
        return []

    records = qdrant.client.retrieve(
        collection_name=qdrant.FACES_COLLECTION,
        ids=[str(f.id) for f in faces],
        with_vectors=True,
    )
    vec_by_id = {r.id: r.vector for r in records}

    usable, vectors = [], []
    for face in faces:
        v = vec_by_id.get(str(face.id))
        if v is not None:
            usable.append(face)
            vectors.append(v)
    if len(usable) < MIN_CLUSTER_SIZE:
        return []

    labels, probs = cluster_embeddings(np.array(vectors))

    def identity_of(face: Face) -> tuple[str, uuid.UUID]:
        return ("person", face.person_id) if face.person_id else ("cluster", face.cluster_id)

    # HDBSCAN kumesi -> icinde gecen kimlikler
    groups: dict[int, dict] = defaultdict(lambda: {"identities": defaultdict(int), "probs": []})
    for face, label, prob in zip(usable, labels, probs):
        if label == -1:
            continue  # gurultu: mevcut klasorunde kalir, onerilere karismaz
        groups[int(label)]["identities"][identity_of(face)] += 1
        groups[int(label)]["probs"].append(float(prob))

    blocked = _blocked_pairs(db)
    suggestions = []
    for label, info in groups.items():
        idents = list(info["identities"].keys())
        if len(idents) < 2:
            continue  # tek klasor: zaten dogru gruplanmis, onerecek bir sey yok
        if _is_blocked(idents, blocked):
            continue
        suggestions.append(
            {
                "identities": [_identity_summary(db, kind, ident, info["identities"][(kind, ident)])
                               for kind, ident in idents],
                "shared_faces": sum(info["identities"].values()),
                "confidence": round(sum(info["probs"]) / len(info["probs"]), 3),
            }
        )

    suggestions.sort(key=lambda s: (-s["confidence"], -s["shared_faces"]))
    return suggestions


def _blocked_pairs(db: Session) -> set[frozenset]:
    """cannot_link kisitlarini kimlik ciftlerine cevirir."""
    blocked = set()
    for c in db.query(ClusterConstraint).filter(ClusterConstraint.type == "cannot_link").all():
        fa = db.query(Face).filter(Face.id == c.face_id_a).first()
        fb = db.query(Face).filter(Face.id == c.face_id_b).first()
        if not fa or not fb:
            continue
        ia = ("person", fa.person_id) if fa.person_id else ("cluster", fa.cluster_id)
        ib = ("person", fb.person_id) if fb.person_id else ("cluster", fb.cluster_id)
        if ia[1] and ib[1] and ia != ib:
            blocked.add(frozenset([ia, ib]))
    return blocked


def _is_blocked(identities: list, blocked: set[frozenset]) -> bool:
    for i in range(len(identities)):
        for j in range(i + 1, len(identities)):
            if frozenset([identities[i], identities[j]]) in blocked:
                return True
    return False


def _identity_summary(db: Session, kind: str, ident: uuid.UUID, shared: int) -> dict:
    if kind == "person":
        p = db.query(Person).filter(Person.id == ident).first()
        label = p.display_name if p else "?"
        total = p.face_count if p else 0
    else:
        c = db.query(Cluster).filter(Cluster.id == ident).first()
        label = f"İsimsiz Kişi ({c.size})" if c else "?"
        total = c.size if c else 0

    sample = (
        db.query(Face)
        .filter(Face.person_id == ident if kind == "person" else Face.cluster_id == ident)
        .first()
    )
    return {
        "kind": kind,
        "id": str(ident),
        "label": label,
        "face_count": total,
        "shared_faces": shared,
        "sample_face": (
            {"face_id": str(sample.id), "photo_id": str(sample.photo_id)} if sample else None
        ),
    }

