"""AWS Bedrock embedding saglayici: Titan Text Embeddings V2
(amazon.titan-embed-text-v2:0). ILERI GECIS OPSIYONU - varsayilan degil.

boto3 uzerinden cagrilir (VLM'in Bedrock yoluyla AYNI kimlik zinciri; bkz.
dispatcher.py). Titan:
  - girdi ONEKI KULLANMAZ (e5'ten farkli),
  - istek basina TEK metin alir -> embed_documents dongude cagirir,
  - body: {"inputText", "dimensions", "normalize": true},
  - yanit: {"embedding": [...], "inputTextTokenCount": N},
  - yalnizca 256 | 512 | 1024 boyut destekler.
"""

from __future__ import annotations

import json
import logging

from app.core.settings import settings

logger = logging.getLogger("photoai.embedding")

TITAN_V2_ALLOWED_DIMS = (256, 512, 1024)


class BedrockTitanProvider:
    def __init__(self) -> None:
        self._validate_dim(settings.EMBEDDING_DIM)
        # Agir/opsiyonel import yalnizca bu saglayici secilince.
        import boto3

        self.name = f"bedrock:{settings.AWS_BEDROCK_EMBED_MODEL_ID}"
        self.dim = settings.EMBEDDING_DIM
        self._client = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)
        logger.info("Bedrock Titan embedding hazir: model=%s dim=%d",
                    settings.AWS_BEDROCK_EMBED_MODEL_ID, self.dim)

    @staticmethod
    def _validate_dim(dim: int) -> None:
        if dim not in TITAN_V2_ALLOWED_DIMS:
            raise ValueError(
                f"EMBEDDING_DIM={dim} Titan Text Embeddings V2 tarafindan desteklenmiyor "
                f"(izin verilen: {TITAN_V2_ALLOWED_DIMS}). EMBEDDING_PROVIDER=bedrock "
                f"kullaniliyorsa EMBEDDING_DIM bunlardan biri olmali."
            )

    def _invoke(self, text: str) -> list[float]:
        resp = self._client.invoke_model(
            modelId=settings.AWS_BEDROCK_EMBED_MODEL_ID,
            body=json.dumps({"inputText": text, "dimensions": self.dim, "normalize": True}),
        )
        payload = json.loads(resp["body"].read())
        return payload["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._invoke(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._invoke(text)
