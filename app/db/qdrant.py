# Qdrant vektor veritabani baglantisi (Teknik Tasarim Dokumani Bolum 12)
#
# PostgreSQL kaynak-of-truth'tur; Qdrant yalnizca aranabilir turev veriyi
# (yuz embeddingleri, kimlik merkezleri) tutar. Her Qdrant noktasi
# PostgreSQL'deki bir kayda baglidir.
#
# IDENTITY_POOL_COLLECTION: "kisiler havuzu" - hem isimlendirilmis kisilerin
# (Person) hem henuz isimlendirilmemis klasorlerin (Cluster) merkezini tek
# koleksiyonda tutar; payload'daki "kind" alani (person|cluster) ayirt eder.
# Yeni gelen her yuz, tek bir sorguyla bu havuzun tamamiyla karsilastirilir
# (Bolum 8.3 gercek-zamanli sadelestirmesi - bkz. face_service._assign_or_bucket).

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.settings import settings

FACES_COLLECTION = "faces"
IDENTITY_POOL_COLLECTION = "identity_pool"
EMBEDDING_DIM = 512

# Semantik arama: VLM JSON analizinden uretilen fotograf-basina TEK metin
# embedding'i. Vektor boyutu YUZ koleksiyonlarindan FARKLI - metin embedding
# modeline bagli (settings.EMBEDDING_DIM: local e5-base=768, bedrock
# titan-v2=1024). point_id = str(photo_id); payload yalnizca teshis icin
# (PostgreSQL + photo_embeddings tablosu kaynak-of-truth).
PHOTO_SEMANTIC_COLLECTION = "photo_semantic"

client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)


def ensure_collections() -> None:
    """Koleksiyonlar yoksa olusturur (Bolum 12: yuz koleksiyonlari 512-d,
    Cosine mesafe). SEMANTIC_SEARCH_ENABLED ise 'photo_semantic' koleksiyonu
    da (settings.EMBEDDING_DIM boyutunda) saglanir."""
    existing = {c.name for c in client.get_collections().collections}
    for name in (FACES_COLLECTION, IDENTITY_POOL_COLLECTION):
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
    if settings.SEMANTIC_SEARCH_ENABLED:
        _ensure_photo_semantic_collection(existing)


def _ensure_photo_semantic_collection(existing: set[str]) -> None:
    """'photo_semantic' koleksiyonunu saglar. ZATEN VARSA ve mevcut vektor
    boyutu settings.EMBEDDING_DIM'e UYMUYORSA - embedding provider/model
    degismis demektir - net bir hatayla BASLANGICTA durdurur (yanlis
    boyutlu vektor upsert'i Qdrant tarafinda sessizce reddedilir/bozulur):
    koleksiyonu dusurup scripts/backfill_photo_semantic.py'yi yeniden
    calistirmak gerekir."""
    want = settings.EMBEDDING_DIM
    if PHOTO_SEMANTIC_COLLECTION not in existing:
        client.create_collection(
            collection_name=PHOTO_SEMANTIC_COLLECTION,
            vectors_config=VectorParams(size=want, distance=Distance.COSINE),
        )
        return
    have = client.get_collection(PHOTO_SEMANTIC_COLLECTION).config.params.vectors.size
    if have != want:
        raise RuntimeError(
            f"Qdrant '{PHOTO_SEMANTIC_COLLECTION}' koleksiyonunun vektor boyutu {have}, "
            f"ayar (EMBEDDING_DIM) {want}. Embedding provider/model degismis olabilir. "
            f"Cozum: koleksiyonu dusurun ve scripts/backfill_photo_semantic.py'yi "
            f"yeniden calistirin."
        )
