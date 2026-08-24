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

client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)


def ensure_collections() -> None:
    """Koleksiyonlar yoksa olusturur (Bolum 12: 512-d, Cosine mesafe)."""
    existing = {c.name for c in client.get_collections().collections}
    for name in (FACES_COLLECTION, IDENTITY_POOL_COLLECTION):
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
