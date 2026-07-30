import base64
import io
import re
import zlib
import httpx
from pydantic import BaseModel, ValidationError, field_validator
from typing import Literal
from app.config import settings
from PIL import Image, ImageOps
import pillow_heif
pillow_heif.register_heif_opener()

# 1024 uzun kenar -> num_ctx=8192 icinde bol bol yer kalir (context tasmasi
# hem HTTP 400'e hem de sessiz dejenere @@@@ ciktisina yol aciyordu).
MAX_DIMENSION = 1024
NUM_CTX = 8192

# Ollama ve LM Studio farkli API formatlari kullaniyor:
#   - Ollama:     /api/chat          , content icinde ayri "images" alani
#   - LM Studio:  /v1/chat/completions (OpenAI-uyumlu), content icinde
#                 image_url bloklari, "options" alanini tanimiyor.
# NOT: LM Studio'da context uzunlugu (num_ctx karsiligi) API'den degil,
# modeli LM Studio arayuzunde yuklerken "Context Length" ayarindan verilir.
IS_LM_STUDIO = settings.AI_PROVIDER == "lm_studio"


def _chat_url() -> str:
    base = settings.VLM_BASE_URL.rstrip("/")
    if IS_LM_STUDIO:
        # settings.VLM_BASE_URL zaten ".../v1" ile bitiyor olmali
        if not base.endswith("/v1"):
            base = base + "/v1"
        return f"{base}/chat/completions"
    else:
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return f"{base}/api/chat"


PROMPT = """Bu fotoğrafı analiz et ve SADECE aşağıdaki JSON şemasında yanıt ver, başka hiçbir metin ekleme:

{
  "caption": "kısa açıklama",
  "environment": "indoor" veya "outdoor" veya "mixed",
  "activity": "yapılan aktivite",
  "people_count": sayı,
  "possible_event": "olası etkinlik türü",
  "summary": "1-2 cümlelik özet"
}

Önemli: people_count HER ZAMAN bir tam sayı olmalı (örnek: 3, 15, 40). Kalabalık veya net sayılamıyorsa yaklaşık bir tam sayı tahmin et (örn. 50). Asla "çok", "birçok", "hundreds" gibi bir kelime yazma."""


class VLMResult(BaseModel):
    caption: str
    environment: Literal["indoor", "outdoor", "mixed"]
    activity: str
    people_count: int
    possible_event: str
    summary: str

    @field_validator("people_count", mode="before")
    @classmethod
    def _coerce_people_count(cls, v):
        # Model bazen "500-600" gibi bir aralik yaziyor; ortalamasini al.
        if isinstance(v, str):
            nums = [int(n) for n in re.findall(r"\d+", v)]
            if nums:
                return round(sum(nums) / len(nums))
        return v


_PLACEHOLDERS = {
    "kısa açıklama",
    "yapılan aktivite",
    "olası etkinlik türü",
    "1-2 cümlelik özet",
}


def _is_placeholder_echo(result: VLMResult) -> bool:
    """Model, prompt'taki örnek metni olduğu gibi geri döndürdüyse tespit eder."""
    fields = [result.caption, result.activity, result.possible_event, result.summary]
    return sum(1 for f in fields if f.strip().lower() in _PLACEHOLDERS) >= 2


def _encode_image(image_path: str) -> str:
    with Image.open(image_path) as img:
        # EXIF rotasyonunu piksele uygula (yoksa yan yatmis fotolar goruluyor)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()


def _clean_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _build_payload(b64: str) -> dict:
    if IS_LM_STUDIO:
        return {
            "model": settings.VLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
            "stream": False,
            "temperature": 0.1,
            # LM Studio "options"/"repeat_penalty" tanimiyor; OpenAI-uyumlu
            # alanlar bunlar. repeat_penalty karsiligi yok, elde degil.
        }
    else:
        return {
            "model": settings.VLM_MODEL,
            "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
            "stream": False,
            "options": {
                "num_ctx": NUM_CTX,
                "temperature": 0.1,
                "repeat_penalty": 1.05,  # dejenere tekrari (@@@@) frenler
            },
        }


def _looks_degenerate(text: str, min_len: int = 150, max_ratio: float = 0.25) -> bool:
    """
    Tekrar dongusunu (karakter VEYA cumle/kelime seviyesinde) yakalar.
    Tekli karakter tekrari (@@@@) ayri kontrolde yakalaniyor; burada
    zlib sikistirma orani kullanilir -- dogal metin ~0.4-0.7 sikisirken
    ayni cumlenin defalarca tekrari ~0.03-0.1'e duser (test ile dogrulandi).
    """
    text = text.strip()
    if len(text) < min_len:
        return False
    raw_bytes = text.encode("utf-8", "ignore")
    ratio = len(zlib.compress(raw_bytes, 9)) / len(raw_bytes)
    return ratio < max_ratio


async def _call_vlm(client: httpx.AsyncClient, payload: dict) -> str:
    r = await client.post(_chat_url(), json=payload)
    if r.status_code != 200:
        print("VLM HATASI:", r.status_code, r.text)
    r.raise_for_status()

    data = r.json()

    if IS_LM_STUDIO:
        choices = data.get("choices") or []
        raw = (choices[0].get("message") or {}).get("content", "") if choices else ""
        # SAGLIK KONTROLU: OpenAI-uyumlu ucta prompt_eval_count yok, onun
        # yerine usage.completion_tokens'in dolu gelmesi bekleniyor.
        usage = data.get("usage") or {}
        if not usage.get("completion_tokens"):
            raise ValueError(f"anormal vlm cevabi (usage.completion_tokens yok): {raw[:80]!r}")
    else:
        raw = (data.get("message") or {}).get("content", "")
        # SAGLIK KONTROLU: prompt_eval_count yoksa uretim normal yolla olmadi
        # (context tasmasinda dejenere ciktida bu alan gelmiyordu).
        if data.get("prompt_eval_count") is None:
            raise ValueError(f"anormal vlm cevabi (prompt_eval_count yok): {raw[:80]!r}")

    # Dejenere tekrar dedektoru: cikti tek/iki karakterin tekrarindan olusuyorsa
    if len(set(raw.strip())) <= 2 and len(raw.strip()) > 20:
        raise ValueError(f"dejenere cikti (karakter tekrari): {raw[:40]!r}")

    if _looks_degenerate(raw):
        raise ValueError(f"dejenere cikti (cumle/kelime dongusu): {raw[:80]!r}")

    return raw


async def analyze_photo(image_path: str) -> VLMResult:
    b64 = _encode_image(image_path)
    payload = _build_payload(b64)

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            raw = await _call_vlm(client, payload)
        result = VLMResult.model_validate_json(_clean_json(raw))
        if _is_placeholder_echo(result):
            raise ValueError("VLM sablonu aynen geri dondurdu")
        return result
    except (ValidationError, ValueError):
        # Bozuk/şablon/dejenere içerikten sonra temiz bir retry
        async with httpx.AsyncClient(timeout=45) as client:
            raw2 = await _call_vlm(client, payload)

        result2 = VLMResult.model_validate_json(_clean_json(raw2))
        if _is_placeholder_echo(result2):
            raise ValueError("VLM sablonu tekrar ediyor, gecerli analiz uretilemedi")
        return result2