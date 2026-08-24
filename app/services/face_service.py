"""Yuz tespiti + hizalama + embedding orkestrasyonu (A5-A8, Bolum 10.1)."""

import json
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
from PIL import Image, ImageOps
from qdrant_client.models import PointStruct
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.ai.face_detector import FaceDetector
from app.ai.face_embedder import AUTO_ASSIGN_THRESHOLD, FaceEmbedder
from app.core.settings import settings
from app.db import identity_locks, qdrant
from app.db.models import Cluster, Face, Photo
from app.services import candidate_search

pillow_heif.register_heif_opener()

FACE_CROPS_DIR = Path("uploads/faces")
FACE_CROPS_DIR.mkdir(parents=True, exist_ok=True)

# --- Arka plan yuzu esikleri (Bolum 8.1) ---
#
# Dokuman: "alani goruntunun %0,1'inden kucuk yuzler 'arka plan yuzu' olarak
# isaretlenir ve kumelemeye dusuk oncelikle girer."
#
# DOKUMANDAN BILINCLI SAPMA: Dokumanin kurali GORELI (goruntu boyutunun yuzdesi).
# Bu, ~1-2 MP fotograflar icin mantikli (%0.1 = ~30x30 yuz) ama modern 50 MP
# telefon fotograflarinda %0.1 = ~223x223 yuze denk geliyor - yani 109x142'lik,
# 0.92 guvenli, net taninabilir bir yuz bile "arka plan" damgasi yiyordu
# (foto_sahne/test7.jpg ile olculdu).
#
# Cozum: iki kosul BIRDEN aranir.
#   (a) goreli: goruntunun %0.1'inden kucuk  -> "sahnenin arka planinda"
#   (b) mutlak: 112x112'den kucuk            -> "embedding'i gercekten guvenilmez"
# 112x112, AuraFace'in girdi boyutu; bunun uzerindeki bir kirpim modelin dogal
# cozunurlugunde demektir, buyutme kaynakli bilgi kaybi yoktur.
#
# Neden tutucu (AND) taraf secildi: yanlis "arka plan" damgasi sessiz veri
# kaybidir (kullanici o kisiyi hic goremez); yanlis "normal" ise yalnizca
# fazladan bir klasordur ve silme butonuyla temizlenebilir.
#
# Her iki deger de diger esikler gibi POC ile kalibre edilecek.
BACKGROUND_FACE_AREA_RATIO = 0.001  # %0.1 (goreli)
BACKGROUND_FACE_MIN_PIXELS = 112 * 112  # AuraFace girdi boyutu (mutlak taban)

# --- Hibrit atama karar loglari (config.HYBRID_*, bkz. _assign_or_bucket) ---
#
# Gorev kisiti geregi veritabani semasina DOKUNULMUYOR - bu yuzden ayri bir
# Postgres tablosu yerine duz JSONL (satir basi bir JSON kaydi) dosyasi
# kullaniliyor. Golge modda (varsayilan) her yuz icin bir satir yazilir;
# "gunler/haftalar sonra golge modda ne siklikla farkli karar verildi, o
# farkli kararlar isabetli miydi" sorusu bu dosya elle ya da bir script ile
# taranarak (ornegin pandas.read_json(path, lines=True)) cevaplanabilir.
HYBRID_LOG_PATH = Path("logs/hybrid_assignment_decisions.jsonl")

_detector: FaceDetector | None = None
_embedder: FaceEmbedder | None = None


def _get_detector() -> FaceDetector:
    global _detector
    if _detector is None:
        _detector = FaceDetector(model_path=settings.YUNET_MODEL_PATH)
    return _detector


def _get_embedder() -> FaceEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = FaceEmbedder(model_dir=settings.AURAFACE_MODEL_DIR)
    return _embedder


def _read_image_bgr(path: str) -> np.ndarray:
    """PIL + pillow_heif ile okur (cv2.imread HEIC'i desteklemez) ve EXIF
    rotasyonunu uygular (yoksa telefon fotograflarinda yuz/landmark yanlis
    cikar) - projenin geri kalaninda zaten kullanilan desen (bkz. dispatcher.py,
    photo_service.get_servable_file)."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _quality_score(image_bgr, bbox: tuple[int, int, int, int], det_confidence: float) -> float:
    """Tespit guveni + bulaniklik (Laplacian varyansi) bilesik kalite skoru (Bolum 8.2)."""
    x, y, w, h = bbox
    crop = image_bgr[max(y, 0): y + h, max(x, 0): x + w]
    if crop.size == 0:
        return det_confidence
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(blur_variance / 500.0, 1.0)  # deneysel normalize, POC ile kalibre edilecek
    return float((det_confidence + blur_score) / 2)


def _decide_assignment(
    db: Session, face_id: uuid.UUID, embedding: np.ndarray, local_pool: dict
) -> tuple[str | None, str | None]:
    """FAZ 1 (bkz. detect_and_embed): SADECE hangi kimlige (person/cluster)
    atanacagina KARAR verir - HICBIR PG/Qdrant YAZIMI yapmaz, HICBIR KILIT
    ALMAZ, TRANSACTION DISINDA cagrilir.

    ONEMLI (revizyon turunde netlestirildi): burada donen (kind, id) SADECE
    "HANGI kimlik" bilgisidir - o kimligin O ANKI centroid/face_count
    DEGERI bu fonksiyondan asla disari sizmaz. Deger, FAZ 2'de
    (_apply_assignment_locked) ilgili satir ZATEN KILITLIYKEN PG'den TAZE
    okunur. Bu ayrim BILINCLI: aramanin dondugu skor/vektor bir ANLIK
    GORUNTUdur, iki cagri arasinda (FAZ 1 -> FAZ 2) baska bir islem ayni
    kimligi guncellemis olabilir - kilit SADECE deger FAZ 2'de taze
    okunursa bir ise yarar (bkz. test_assign_or_bucket_pr_c.py::
    test_locked_update_uses_fresh_value_not_stale_search_snapshot).

    local_pool: bu fotografin BU TURDA (henuz PG/Qdrant'a yazilmamis)
    kararlastirdigi YENI kumeleri tutan, cagiranin (detect_and_embed FAZ 1
    donguсu) yonettigi bellek-ici sozluk (id -> {kind, centroid, count}).
    Bu, reddedilen surec-ici onbellek DEGIL: kapsami TEK bir handler
    cagrisi/TEK fotograf, hicbir sekilde cagrilar arasi paylasilmiyor,
    disaridan gorunmuyor. Amac: OLCULEN gercek durum - ayni fotografta ayni
    kisinin >1 yuzu vakalarin %4.87'sinde (coklu-yuzlu fotograflarin
    %8.84'unde) gorulmus - bu kisi sistemde ILK KEZ ortaya cikiyorsa,
    local_pool OLMADAN Faz 1'deki DB/Qdrant aramasi bu fotografin bir
    ONCEKI yuzunun (henuz commit edilmemis) actigi kumeyi GOREMEZ, IKI AYRI
    BUCKET acilir. local_pool bu adaylari da arama sonucuna KATAR - Faz
    2'deki kilit/dogruluk garantisi DEGISMEZ (deger yine PG'den taze
    okunur), sadece "hangi kimlik" karari daha dogru verilir.

    Donus: (final_kind, final_id) - final_kind None ise eslesme yok
    (cagiran, arka plan degilse yeni klasor acar).
    """
    compute_hybrid = settings.HYBRID_ASSIGNMENT_ENABLED or settings.HYBRID_SHADOW_MODE
    query_limit = max(settings.HYBRID_TOP_CANDIDATES, 1) if compute_hybrid else 1

    db_candidates = candidate_search.get_candidate_finder().find_candidates(embedding, query_limit)

    # local_pool'daki (bu fotografta YENI acilmis, henuz PG'de olmayan)
    # kumeleri de aday listesine kat. NOT: _hybrid_decision'in "gri bolge"
    # dalindaki _best_neighbor_match, PG'den commit edilmis Face satiri
    # arar - local_pool adaylarinin HENUZ hic committed uyesi olmadigi icin
    # o dal onlari GOREMEZ (skor -1.0 doner). Bu, HYBRID_ASSIGNMENT_ENABLED
    # VARSAYILAN OLARAK KAPALI (shadow-mode bile gercek karari etkilemez)
    # oldugu icin bilinen, kabul edilmis bir sinir - asagidaki Yontem A
    # (aktif karar yolu) local_pool adaylarini TAM olarak kullanir.
    normalized_embedding = embedding / np.linalg.norm(embedding)
    local_candidates = [
        candidate_search.Candidate(
            score=float(np.dot(normalized_embedding, entry["centroid"])),
            payload={"kind": entry["kind"], "id": ident, "face_count": entry["count"]},
            vector=entry["centroid"].tolist(),
        )
        for ident, entry in local_pool.items()
    ]
    candidates = sorted(db_candidates + local_candidates, key=lambda c: -c.score)[:query_limit]

    method_a_kind, method_a_id, method_a_score = None, None, 0.0
    if candidates and candidates[0].score >= AUTO_ASSIGN_THRESHOLD:
        payload = candidates[0].payload or {}
        method_a_kind = payload.get("kind")
        method_a_id = payload.get("id")
        method_a_score = candidates[0].score

    final_kind, final_id, final_score = method_a_kind, method_a_id, method_a_score

    if compute_hybrid:
        hybrid_kind, hybrid_id, hybrid_score, branch = _hybrid_decision(db, embedding, candidates)
        decisions_differ = (method_a_kind, method_a_id) != (hybrid_kind, hybrid_id)
        _log_hybrid_decision(
            face_id=face_id,
            branch=branch,
            method_a_kind=method_a_kind,
            method_a_id=method_a_id,
            method_a_score=method_a_score,
            hybrid_kind=hybrid_kind,
            hybrid_id=hybrid_id,
            hybrid_score=hybrid_score,
            decisions_differ=decisions_differ,
        )
        if settings.HYBRID_ASSIGNMENT_ENABLED and not settings.HYBRID_SHADOW_MODE:
            final_kind, final_id, final_score = hybrid_kind, hybrid_id, hybrid_score

    return final_kind, final_id


def _best_neighbor_match(db: Session, embedding: np.ndarray, kind: str, ident: str) -> float:
    """Hibrit gri-bolge karsilastirmasi: bir aday kimligin (person/cluster)
    TUM uyelerini (arka plan yuzleri haric, Bolum 8.1) getirir, yeni gelen
    embedding'i her biriyle tek tek kosinus benzerligiyle karsilastirir,
    HYBRID_QUALITY_WEIGHTING_ENABLED ise komsunun quality_score'uyla (sqrt)
    agirliklandirir, en yuksek (agirlikli) benzerligi dondurur.

    NOT: faces koleksiyonunun Qdrant payload'indaki person_id/cluster_id
    alanlari gercek-zamanli atama (bu fonksiyon) tarafindan hic
    guncellenmiyor - sadece Postgres guncelleniyor (Bolum 11). Bu yuzden
    filtre burada da Postgres uzerinden yapilir, embedding'ler sonra
    Qdrant'tan ID ile cekilir - clustering_service.suggest_merges() ile
    ayni desen.
    """
    filter_kwargs = {"person_id": uuid.UUID(ident)} if kind == "person" else {"cluster_id": uuid.UUID(ident)}
    neighbors = (
        db.query(Face)
        .filter_by(**filter_kwargs)
        .filter(Face.is_background.is_(False))
        .all()
    )
    if not neighbors:
        return -1.0

    records = qdrant.client.retrieve(
        collection_name=qdrant.FACES_COLLECTION,
        ids=[str(f.id) for f in neighbors],
        with_vectors=True,
    )
    vec_by_id = {r.id: np.asarray(r.vector, dtype=np.float64) for r in records}

    best = -1.0
    for neighbor in neighbors:
        neighbor_vec = vec_by_id.get(str(neighbor.id))
        if neighbor_vec is None:
            continue
        sim = float(
            np.dot(embedding, neighbor_vec) / (np.linalg.norm(embedding) * np.linalg.norm(neighbor_vec))
        )
        if settings.HYBRID_QUALITY_WEIGHTING_ENABLED:
            if neighbor.quality_score is not None:
                sim *= float(np.sqrt(max(neighbor.quality_score, 0.0)))
            # quality_score NULL/eksik -> agirliksiz benzerlik (gorev notu)
        if sim > best:
            best = sim
    return best


def _hybrid_decision(
    db: Session, embedding: np.ndarray, candidates: list
) -> tuple[str | None, str | None, float, str]:
    """Yontem B (hibrit): scripts/test_hybrid_assignment.py'de canli veri
    uzerinde leave-one-out simulasyonuyla dogrulanan mantigin (bkz.
    reports/hybrid_test_20260807_094041.json) canli koddaki karsiligi.

    "Net" durum (top1 hem yeterince yuksek hem rakibinden yeterince acik
    farkli) -> Yontem A ile ayni sonuc, ekstra sorgu yok.
    "Gri bolge" (dusuk skor VEYA top1/top2 birbirine cok yakin) -> top-N
    adayin HER birinin uyeleriyle tek tek (kalite agirlikli) karsilastirma
    yapilir; merkezin tek noktaya sikistirdigi bilgiyi (ayni kisinin farkli
    acili/kaliteli fotograflari) geri getirir.

    Donus: (kind, id, score, branch). Eslesme yoksa (None, None, best_score, "gri_bolge").
    """
    if not candidates:
        return None, None, 0.0, "gri_bolge"

    top1_score = candidates[0].score
    top2_score = candidates[1].score if len(candidates) > 1 else 0.0

    is_net = (
        top1_score >= settings.HYBRID_CONFIDENCE_THRESHOLD
        and (top1_score - top2_score) >= settings.HYBRID_GAP_THRESHOLD
    )
    if is_net:
        payload = candidates[0].payload or {}
        return payload.get("kind"), payload.get("id"), top1_score, "net"

    best_kind, best_id, best_score = None, None, -1.0
    for candidate in candidates:
        payload = candidate.payload or {}
        kind, ident = payload.get("kind"), payload.get("id")
        if not kind or not ident:
            continue
        neighbor_score = _best_neighbor_match(db, embedding, kind, ident)
        if neighbor_score > best_score:
            best_kind, best_id, best_score = kind, ident, neighbor_score

    if best_kind is not None and best_score >= AUTO_ASSIGN_THRESHOLD:
        return best_kind, best_id, best_score, "gri_bolge"
    return None, None, max(best_score, 0.0), "gri_bolge"


def _log_hybrid_decision(
    *,
    face_id: uuid.UUID,
    branch: str,
    method_a_kind: str | None,
    method_a_id: str | None,
    method_a_score: float,
    hybrid_kind: str | None,
    hybrid_id: str | None,
    hybrid_score: float,
    decisions_differ: bool,
) -> None:
    """Adim 3 (golge mod izlenebilirligi): her hibrit degerlendirmesini tek
    satirlik JSON olarak HYBRID_LOG_PATH'e ekler. Veritabani semasi
    DEGISTIRILMEDI (gorev kisiti) - bu yuzden ayri bir Postgres tablosu
    yerine duz JSONL dosyasi kullanildi."""
    HYBRID_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "face_id": str(face_id),
        "branch": branch,
        "method_a_decision": {"kind": method_a_kind, "id": method_a_id} if method_a_kind else None,
        "method_a_score": round(float(method_a_score), 4),
        "hybrid_decision": {"kind": hybrid_kind, "id": hybrid_id} if hybrid_kind else None,
        "hybrid_score": round(float(hybrid_score), 4),
        "decisions_differ": decisions_differ,
        "created_at": datetime.utcnow().isoformat(),
    }
    with open(HYBRID_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _apply_assignment_locked(
    db: Session,
    face: Face,
    embedding: np.ndarray,
    final_kind: str | None,
    final_id: str | None,
    is_new_identity: bool,
) -> dict | None:
    """FAZ 2 (bkz. detect_and_embed): FAZ 1'in kararini (final_kind/
    final_id/is_new_identity) PG'ye uygular. CAGIRANIN transaction'inda
    calisir - ilgili kimlik (is_new_identity=False ise) detect_and_embed
    tarafindan BU FONKSIYON CAGRILMADAN ONCE identity_locks.lock_identities
    ile ZATEN KILITLENMIS olmalidir. is_new_identity=True ise satir HENUZ
    YOK - kilit GEREKMEZ (kimse referans veremez), bu fonksiyon satiri
    SIMDI olusturur.

    is_new_identity: FAZ 1'de bu fotografin BU TURDA actigi (henuz PG'de
    olmayan) bir kume icin True - final_kind daima "cluster" olur bu
    durumda (bkz. detect_and_embed local_pool notu). Ayni fotografin
    SONRAKI bir yuzu AYNI yeni kumeye eslesirse (local_pool araciligiyla)
    final_id AYNI kalir ama is_new_identity ARTIK False'tur - o zaman
    normal "mevcut kimlik" yolundan (asagida) gecer; satir bu fonksiyonun
    ONCEKI cagrisinda zaten flush edilmis oldugu icin AYNI transaction
    icinde GORULEBILIR (Postgres read-your-own-writes).

    DEGER (centroid/count), is_new_identity=False durumunda HER ZAMAN
    burada (satir ZATEN KILITLI/goruleBILIR oldugu icin) TAZE okunur - FAZ
    1'in arama sonucu SADECE "hangi kimlik" icin kullanildi, DEGER icin
    degil (bkz. test_locked_update_uses_fresh_value_not_stale_search_snapshot).

    TAZE SELECT ayrica kimligin HALA AKTIF oldugunu dogrular (deleted_at IS
    NULL / status='unlabeled') - degilse identity_locks.StaleIdentityDecision
    firlatir (bkz. o sinifin docstring'i): FAZ 1 ile FAZ 2 arasinda baska
    bir islem (merge/delete/label) bu kimligi RETIRE etmis demektir, GERCEK
    bir hata DEGIL.

    BILINEN, KOD DEGISIKLIGI GEREKTIRMEYEN ACIK (PR-F'e birakildi): bu
    kontrol FAZ 2'yi korur ama FAZ 3'u (commit SONRASI Qdrant dual-write,
    kilit DISINDA) korumaz - FAZ 2 basariyla commit olduktan HEMEN SONRA,
    FAZ 3'un Qdrant cagrisindan ONCE bir silme/merge araya girip Qdrant
    identity_pool noktasini silerse, bu FAZ 3 cagrisi o noktayi YENIDEN
    yaratabilir (dar ama sifir olmayan bir pencere - Qdrant'in PG ile
    paylasilan bir kilidi yok). Bu, PR-F'nin (periyodik mutabakat) kapsamina
    BILINCLI olarak birakildi.

    Qdrant'a YAZMAZ (I/O yok, kilitli transaction icinde YASAK) - donen
    dict (None degilse), cagiranin commit SONRASI (FAZ 3) uygulamasi
    icindir.
    """
    if final_kind is None:
        # SADECE arka plan yuzu + hicbir eslesme (FAZ 1'in local_pool dahil
        # aramasi da bos donmus) - kimliksiz havuzda bekler, yeni klasor
        # ACILMAZ (Bolum 8.3 gurultu davranisi).
        return None

    identity_uuid = uuid.UUID(final_id)
    if final_kind == "person":
        face.person_id = identity_uuid
    else:
        face.cluster_id = identity_uuid
    face.assigned_by = "auto"

    if face.is_background:
        # Atandi ama merkeze KATKI VERMEZ (dusuk oncelik). Sayac da
        # artmaz: face_count merkezin kac guvenilir yuzden hesaplandigini
        # tutar, artimli ortalamanin agirligi buna dayanir.
        return None

    if is_new_identity:
        # Bu fotografta YENI acilan kume - satiri SIMDI olustur. ID FAZ
        # 1'de ONCEDEN uretilmisti (ayni fotografin sonraki bir yuzu
        # local_pool araciligiyla AYNI final_id'ye eslesebilsin diye).
        normalized_embedding = embedding / np.linalg.norm(embedding)
        cluster = Cluster(
            id=identity_uuid,
            status="unlabeled",
            size=1,
            created_at=datetime.utcnow(),
            centroid=normalized_embedding.astype(np.float32).tobytes(),
            centroid_updated_at=datetime.utcnow(),
        )
        db.add(cluster)
        db.flush()
        return {
            "identity_id": final_id,
            "centroid": normalized_embedding.tolist(),
            "payload": {"kind": "cluster", "id": final_id, "face_count": 1},
        }

    table = "persons" if final_kind == "person" else "clusters"
    count_col = "face_count" if final_kind == "person" else "size"
    # Kimlik "hala aktif" mi kontrolu (revizyon turu - Madde 6/9 bulgusu):
    # FAZ 1'in karari VERILDIGI anda dogruydu ama FAZ 1 ile FAZ 2 arasinda
    # baska bir islem (merge/delete/label) bu kimligi RETIRE etmis olabilir.
    # persons.deleted_at / clusters.status ZATEN VAR olan alanlar - satir
    # hala VAR ama artik gecerli bir yazma hedefi DEGIL. Filtre uygulanmazsa
    # (eski davranis), silinmis/merge edilmis bir kimlik SESSIZCE dirilir
    # ("hayalet kayit").
    liveness_clause = "AND deleted_at IS NULL" if final_kind == "person" else "AND status = 'unlabeled'"

    # Satir ZATEN kilitli (identities_to_lock) YA DA bu fotografin ONCEKI
    # bir yuzu tarafindan bu transaction icinde ZATEN flush edilmis
    # (is_new_identity=False ama yine de local_pool kaynakli) - iki
    # durumda da AYRICA FOR UPDATE gerekmiyor, TAZE deger okumak yeterli.
    row = db.execute(
        text(f"SELECT centroid, {count_col} FROM {table} WHERE id = :id {liveness_clause}"),
        {"id": identity_uuid},
    ).fetchone()

    if row is None:
        # Kimlik ARTIK gecerli degil (soft-delete/merge/label edilmis) -
        # FAZ 1'in karari GECERSIZ hale gelmis. GERCEK bir hata DEGIL - bkz.
        # identity_locks.StaleIdentityDecision docstring'i.
        raise identity_locks.StaleIdentityDecision(
            f"Kimlik artik gecerli degil (silinmis/merge/label edilmis): kind={final_kind} id={final_id}"
        )

    if row[0] is not None:
        old_centroid = np.frombuffer(bytes(row[0]), dtype=np.float32).astype(np.float64)
        old_count = row[1]
    else:
        # PG kolonu henuz doldurulmamis (ilk backfill/cutover ARASI gecis
        # penceresi) - AKTIF arama backend'inden GERI DUSULUR (guvenlik
        # agi, normal akiste devreye girmez).
        old_centroid = candidate_search.get_candidate_finder().fetch_vector(final_kind, final_id)
        old_count = row[1]

    new_centroid = old_centroid * old_count + embedding
    new_centroid = new_centroid / np.linalg.norm(new_centroid)
    new_count = old_count + 1

    db.execute(
        text(
            f"UPDATE {table} SET centroid = :c, {count_col} = :n, "
            f"centroid_updated_at = now() WHERE id = :id"
        ),
        {"c": new_centroid.astype(np.float32).tobytes(), "n": new_count, "id": identity_uuid},
    )

    return {
        "identity_id": final_id,
        "centroid": new_centroid.tolist(),
        "payload": {"kind": final_kind, "id": final_id, "face_count": new_count},
    }


def get_faces_for_photo(db: Session, photo_id: uuid.UUID) -> list[Face]:
    return db.query(Face).filter(Face.photo_id == photo_id).all()


def get_face(db: Session, face_id: uuid.UUID) -> Face | None:
    return db.query(Face).filter(Face.id == face_id).first()


def detect_and_embed(db: Session, photo: Photo) -> list[Face]:
    """Bir fotograftaki tum yuzleri tespit eder, hizalar, embed eder ve kaydeder.

    Sirasi (Bolum 10.1): A5 tespit -> A6 hizalama/kirpim -> A7 embedding ->
    A8 kimlik atama denemesi.

    UC FAZLI YAPI (revizyon turunde zorunlu kilindi - onceki tasarim, her
    yuzun centroid guncellemesini KENDI ayri/hemen-committed transaction'inda
    yapiyordu; bu, fotografin SONRAKI bir yuzu hata verip bu DIS transaction
    rollback olursa, ONCEKI yuzlerin centroid katkisinin PG'de KALICI kalip
    Face satirinin HIC olusmamasina yol aciyordu - "anna" bug sinifinin
    mimarinin DUZENLI URETTIGI bir sonucu haline gelmesi, PR-A'nin "ayni
    satir/ayni kilit/tek sayac" tasarim gerekcesini GECERSIZ kilan bir durum):

      FAZ 1 (agir is, TRANSACTION/KILIT YOK): tespit, hizalama, embedding,
        HER yuz icin aday arama karari (_decide_assignment - SADECE "hangi
        kimlik", DEGER degil). cv2/onnxruntime GPU/CPU isi burada, DB'yi
        bekletmez.
      FAZ 2 (TEK KISA transaction): once bu fotografin dokunacagi (YENI
        acilanlar HARIC - onlarin satiri henuz yok) TUM kimlikler TEK
        SEFERDE kilitlenir (identity_locks.lock_identities - deadlock'suz
        sirali), sonra TUM Face satirlari yazilir + HER yuzun kimlik
        atamasi PG'ye uygulanir (_apply_assignment_locked - deger burada,
        satir kilitli/gorulebilir oldugunda, TAZE okunur), TEK commit.
        Qdrant'a HIC dokunulmaz (kilit altinda I/O YASAK).
      FAZ 3 (commit SONRASI, kilit YOK): Qdrant dual-write - hem
        FACES_COLLECTION (ham yuz vektorleri - bu da PG-once oldu, onceki
        tasarimda commit'ten ONCE yaziliyordu) hem IDENTITY_POOL_COLLECTION
        (GECIS DONEMI ayna, bkz. candidate_search.py basi).

    FOTOGRAF-ICI TUTARLILIK (local_pool - onceki turde "birbirini goremez"
    diye kabul edilen davranis DUZELTILDI): olcum, ayni fotografta ayni
    kisinin >1 yuzu gorulme oraninin vakalarin %4.87'si (coklu-yuzlu
    fotograflarin %8.84'u) oldugunu gosterdi - nadir DEGIL. FAZ 1'deki HER
    _decide_assignment cagrisina, bu fotografin BU TURDA actigi (henuz
    PG'de olmayan) kumeleri de aday olarak sunan bir local_pool (bellek-ici
    sozluk, kapsami TEK bu fonksiyon cagrisi - reddedilen surec-ici
    onbellekle KARISTIRILMASIN) esliK eder. Boylece ayni fotografta ilk kez
    ortaya cikan bir kisinin iki yuzu TEK kumede toplanir, Faz 2'nin kilit/
    dogruluk garantisi DEGISMEDEN.
    """
    image_bgr = _read_image_bgr(photo.storage_path)
    image_area = image_bgr.shape[0] * image_bgr.shape[1]

    detector = _get_detector()
    embedder = _get_embedder()

    # --- FAZ 1: agir is + karar (+ fotograf-ici local_pool), TRANSACTION/KILIT YOK ---
    prepared: list[dict] = []
    local_pool: dict[str, dict] = {}  # id(str) -> {"kind","centroid","count"}
    for detected in detector.detect(image_bgr):
        embedding, aligned_crop = embedder.get_embedding(image_bgr, detected.landmarks)

        _, _, bw, bh = detected.bbox
        face_pixels = bw * bh
        is_background = (
            face_pixels / image_area < BACKGROUND_FACE_AREA_RATIO
            and face_pixels < BACKGROUND_FACE_MIN_PIXELS
        )
        quality_score = _quality_score(image_bgr, detected.bbox, detected.confidence)
        face_id = uuid.uuid4()  # Face satiri FAZ 2'de olusacak; ID'yi simdiden
        # sabitliyoruz (hibrit karar loglamasi face_id ister, FAZ 1'de henuz
        # PG satiri yok).

        final_kind, final_id = _decide_assignment(db, face_id, embedding, local_pool)

        is_new_identity = False
        if final_kind is None and not is_background:
            # Havuzda (DB + local_pool) eslesme yok, guvenilir bir yuz ->
            # yeni, tek yuzluk isimsiz klasor acilacak. ID'yi SIMDI uret ki
            # bu fotografin SONRAKI bir yuzu (varsa) local_pool araciligiyla
            # AYNI kumeye eslesebilsin.
            new_cluster_id = uuid.uuid4()
            final_kind, final_id = "cluster", str(new_cluster_id)
            is_new_identity = True
            normalized = embedding / np.linalg.norm(embedding)
            local_pool[final_id] = {"kind": "cluster", "centroid": normalized, "count": 1}
        elif final_id is not None and final_id in local_pool and not is_background:
            # Bu fotografta YENI acilmis bir kumeye eslesti - yerel merkezi
            # GUNCELLE (sonraki bir yuz daha da GUNCEL bir goruntu gorsun).
            # GERCEK bir DB kimligine eslesenler local_pool'a HIC girmez -
            # Faz 2 zaten PG'den taze okuyacak, burada izlemeye gerek yok.
            prev = local_pool[final_id]
            updated_centroid = prev["centroid"] * prev["count"] + embedding
            updated_centroid = updated_centroid / np.linalg.norm(updated_centroid)
            local_pool[final_id] = {
                "kind": final_kind, "centroid": updated_centroid, "count": prev["count"] + 1,
            }

        prepared.append({
            "face_id": face_id,
            "detected": detected,
            "embedding": embedding,
            "aligned_crop": aligned_crop,
            "is_background": is_background,
            "quality_score": quality_score,
            "final_kind": final_kind,
            "final_id": final_id,
            "is_new_identity": is_new_identity,
        })

    # --- FAZ 2: TEK KISA transaction - kilit + Face satirlari + centroid ---
    # YENI acilan kimlikler (is_new_identity=True) kilitlenmez - satirlari
    # HENUZ yok, kimse referans veremez (bkz. _apply_assignment_locked).
    identities_to_lock = sorted({
        (p["final_kind"], uuid.UUID(p["final_id"]))
        for p in prepared
        if p["final_kind"] is not None and not p["is_background"] and not p["is_new_identity"]
    })
    identity_locks.lock_identities(db, identities_to_lock)

    saved_faces: list[Face] = []
    pending_face_vectors: list[dict] = []
    pending_identity_ops: list[dict] = []

    for p in prepared:
        detected = p["detected"]
        face = Face(
            id=p["face_id"],
            photo_id=photo.id,
            bbox={
                "x": detected.bbox[0],
                "y": detected.bbox[1],
                "w": detected.bbox[2],
                "h": detected.bbox[3],
            },
            landmarks=[list(point) for point in detected.landmarks],
            det_confidence=detected.confidence,
            quality_score=p["quality_score"],
            is_background=p["is_background"],
            crop_path="",  # asagida doldurulur
        )
        db.add(face)
        db.flush()

        crop_path = FACE_CROPS_DIR / f"{face.id}.jpg"
        cv2.imwrite(str(crop_path), p["aligned_crop"])
        face.crop_path = str(crop_path)

        pending_face_vectors.append({
            "face_id": str(face.id), "embedding": p["embedding"], "quality_score": face.quality_score,
        })

        identity_op = _apply_assignment_locked(
            db, face, p["embedding"], p["final_kind"], p["final_id"], p["is_new_identity"],
        )
        if identity_op is not None:
            pending_identity_ops.append(identity_op)

        saved_faces.append(face)

    db.commit()
    for face in saved_faces:
        db.refresh(face)

    # --- FAZ 3: Qdrant dual-write, commit SONRASI (kilit YOK, PG-once kurali) ---
    for op in pending_face_vectors:
        qdrant.client.upsert(
            collection_name=qdrant.FACES_COLLECTION,
            points=[
                PointStruct(
                    id=op["face_id"],
                    vector=op["embedding"].tolist(),
                    payload={
                        "photo_id": str(photo.id),
                        "person_id": None,
                        "cluster_id": None,
                        "quality_score": op["quality_score"],
                        "created_at": photo.created_at.isoformat() if photo.created_at else None,
                    },
                )
            ],
        )
    for op in pending_identity_ops:
        qdrant.client.upsert(
            collection_name=qdrant.IDENTITY_POOL_COLLECTION,
            points=[PointStruct(id=op["identity_id"], vector=op["centroid"], payload=op["payload"])],
        )

    return saved_faces
