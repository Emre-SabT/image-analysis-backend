import base64
import io
import re
import zlib

import boto3
import httpx
from pydantic import BaseModel, ValidationError, field_validator
from typing import Literal
from app.core.settings import settings
from PIL import Image, ImageOps
import pillow_heif
pillow_heif.register_heif_opener()

# 768 uzun kenar (onceki: 1024) - GPU offload ile hiz testinde kucultmenin
# ayrica vision-token sayisini (ve dolayisiyla hem sureyi hem KV-cache VRAM
# ihtiyacini) dusurdugu olculdu. 1024'te tasma sorunu (HTTP 400 / dejenere
# @@@@ ciktisi) 768'de de gecerliligini koruyor, bu yuzden ust sinir olarak
# tutuluyor.
MAX_DIMENSION = 768

# AI_PROVIDER'a gore iki cok farkli cagri yolu var:
#   - lm_studio: yerel, OpenAI-uyumlu HTTP (/v1/chat/completions), httpx ile.
#   - bedrock:   AWS Bedrock, boto3 ile (HTTP degil, AWS SDK cagrisi).
# NOT (degisim gecmisi): Ollama destegi kaldirildi, yerine AWS Bedrock
# eklendi - kullanicinin acik istegiyle.
IS_LM_STUDIO = settings.AI_PROVIDER == "lm_studio"


def _chat_url() -> str:
    """Sadece LM Studio icin - Bedrock HTTP degil, boto3 uzerinden cagrilir."""
    base = settings.VLM_BASE_URL.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    return f"{base}/chat/completions"


PROMPT = """Sen bir görsel analiz ve medya metadata üretim uzmanısın. Bu fotoğrafı analiz et ve SADECE aşağıdaki JSON şemasında yanıt ver, başka hiçbir metin ekleme.

Kurallar:
- Tüm metin alanları HER ZAMAN Türkçe olmalı. Görselde İngilizce yazı, marka adı veya metin görsen bile açıklamaları Türkçe yaz. İngilizce kelime veya cümle kullanma.
- Sadece fotoğrafta açıkça görülen veya güçlü şekilde çıkarılabilen bilgileri yaz; tahmine dayalı uydurma detay ekleme.
- Gereksiz tekrar yapma, çok genel/boş etiketler ("fotoğraf", "görsel" gibi) kullanma.
- description 1-2 cümle olsun.
- primary_object, action, mood, use_case, possible_event tek string olmalı.
- secondary_objects, environment, attributes, context, style, audience array olmalı; bilgi yoksa boş dizi döndür.
- people_count HER ZAMAN bir tam sayı olmalı (örnek: 0, 3, 15, 40). Kalabalık veya net sayılamıyorsa yaklaşık bir tam sayı tahmin et (örn. 50). Asla "çok", "birçok", "hundreds" gibi bir kelime yazma.
- public_figures sadece tanınmış veya kamusal açıdan önemli kişiler için doldurulmalı (sıradan/özel kişiler için değil). Bir kişiden emin değilsen name alanını boş string ("") yap, types alanını yine de doldurabilirsin. Aynı kişi tekrar etmemeli.
- all_tags, description ve public_figures.name HARİÇ tüm alanların birleşiminden oluşmalı: primary_object, secondary_objects, environment, attributes, action, mood, use_case, context, style, audience, public_figures.types. Duplicate etiket kullanma, all_tags içine açıklama cümlesi ekleme.
- Fotoğrafta ürün üzerinde okunabilir bir marka/metin varsa (ör. ambalaj, etiket, kutu üzerindeki yazı), açıklamanı o yazıya dayandır; yazıyla çelişen bir varsayımda bulunma (ör. kutunun üzerinde "kahve" yazıyorsa "bira" deme).

JSON şeması:
{
  "description": "kısa açıklama (1-2 cümle)",
  "environment_type": "indoor" veya "outdoor" veya "mixed",
  "people_count": sayı,
  "possible_event": "olası etkinlik türü",
  "primary_object": "ana nesne veya odak",
  "secondary_objects": ["ikincil nesneler"],
  "environment": ["ortam/mekan etiketleri"],
  "attributes": ["görsel öznitelikler (renk, ışık, kompozizyon vb.)"],
  "action": "yapılan aktivite",
  "mood": "atmosfer / ruh hali",
  "use_case": "olası kullanım amacı",
  "context": ["bağlamsal etiketler"],
  "style": ["görsel/fotoğraf stili etiketleri"],
  "audience": ["hedef kitle etiketleri"],
  "public_figures": [
    {"name": "isim veya boş string", "types": ["ör. sporcu, sanatçı, siyasetçi"]}
  ],
  "all_tags": ["tüm etiketlerin birleşimi"]
}
"""


class PublicFigure(BaseModel):
    name: str = ""
    types: list[str] = []

    @field_validator("types", mode="before")
    @classmethod
    def _coerce_types(cls, v):
        # Model bazen diziyi "sanatci, oyuncu" gibi virgullu tek string olarak
        # donduruyor; virgulle bolup listeye cevir.
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return v


class VLMResult(BaseModel):
    description: str
    environment_type: Literal["indoor", "outdoor", "mixed"]
    people_count: int
    possible_event: str
    primary_object: str
    secondary_objects: list[str] = []
    environment: list[str] = []
    attributes: list[str] = []
    action: str
    mood: str
    use_case: str
    context: list[str] = []
    style: list[str] = []
    audience: list[str] = []
    public_figures: list[PublicFigure] = []
    all_tags: list[str] = []

    @field_validator("people_count", mode="before")
    @classmethod
    def _coerce_people_count(cls, v):
        # Model bazen "500-600" gibi bir aralik yaziyor; ortalamasini al.
        if isinstance(v, str):
            nums = [int(n) for n in re.findall(r"\d+", v)]
            if nums:
                return round(sum(nums) / len(nums))
        return v


# JSON semasindaki aciklayici placeholder metinler -- model bunlari aynen
# geri dondurursek "sablonu papagan gibi tekrarladi" olarak isaretleriz.
_PLACEHOLDERS = {
    "kısa açıklama (1-2 cümle)",
    "olası etkinlik türü",
    "ana nesne veya odak",
    "yapılan aktivite",
    "atmosfer / ruh hali",
    "olası kullanım amacı",
}


def _is_placeholder_echo(result: VLMResult) -> bool:
    """Model, prompt'taki örnek metni olduğu gibi geri döndürdüyse tespit eder."""
    fields = [
        result.description,
        result.possible_event,
        result.primary_object,
        result.action,
        result.mood,
        result.use_case,
    ]
    return sum(1 for f in fields if f.strip().lower() in _PLACEHOLDERS) >= 2


def _encode_image(image_path: str) -> bytes:
    """JPEG bayt dizisi dondurur (Bedrock'un Converse API'si ham bayt bekler;
    LM Studio icin gerektiginde base64'e ayrica cevrilir - bkz. _call_vlm_lm_studio)."""
    with Image.open(image_path) as img:
        # EXIF rotasyonunu piksele uygula (yoksa yan yatmis fotolar goruluyor)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def _clean_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


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


def _check_degenerate(raw: str) -> None:
    """Her iki saglayici icin ORTAK dejenere-cikti kontrolu. Sorun varsa
    ValueError firlatir (analyze_photo bunu yakalayip TEK bir temiz retry yapar)."""
    if len(set(raw.strip())) <= 2 and len(raw.strip()) > 20:
        raise ValueError(f"dejenere cikti (karakter tekrari): {raw[:40]!r}")
    if _looks_degenerate(raw):
        raise ValueError(f"dejenere cikti (cumle/kelime dongusu): {raw[:80]!r}")


# ---------------------------------------------------------------------------
# LM Studio (yerel, OpenAI-uyumlu HTTP)
# ---------------------------------------------------------------------------


def _build_payload_lm_studio(b64: str) -> dict:
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
        # Onceden hic yoktu - model dejenere/donen bir ciktiya girdiginde
        # (bkz. _check_degenerate) sinirsiz token uretip suresiz uzayabiliyordu.
        # JSON semasi (Bkz. PROMPT) tipik olarak ~200-400 token'a sigar; 700
        # bolluca pay biraktigi halde kacak/donen uretimi erken kesiyor.
        "max_tokens": 700,
        # LM Studio "options"/"repeat_penalty" tanimiyor; OpenAI-uyumlu
        # alanlar bunlar. repeat_penalty karsiligi yok, elde degil.
    }


async def _call_vlm_lm_studio(image_bytes: bytes) -> str:
    payload = _build_payload_lm_studio(base64.b64encode(image_bytes).decode())

    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post(_chat_url(), json=payload)
        if r.status_code != 200:
            print("VLM HATASI:", r.status_code, r.text)
        r.raise_for_status()
        data = r.json()

    choices = data.get("choices") or []
    raw = (choices[0].get("message") or {}).get("content", "") if choices else ""
    # SAGLIK KONTROLU: OpenAI-uyumlu ucta prompt_eval_count yok, onun
    # yerine usage.completion_tokens'in dolu gelmesi bekleniyor.
    usage = data.get("usage") or {}
    if not usage.get("completion_tokens"):
        raise ValueError(f"anormal vlm cevabi (usage.completion_tokens yok): {raw[:80]!r}")

    _check_degenerate(raw)
    return raw


# ---------------------------------------------------------------------------
# AWS Bedrock
# ---------------------------------------------------------------------------

_bedrock_client = None


def _get_bedrock_client():
    """Tek ornek (singleton) - her cagrida yeni boto3 client'i acmamak icin."""
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)
    return _bedrock_client


def _call_vlm_bedrock_sync(image_bytes: bytes) -> str:
    """Bedrock'un birlesik Converse API'sini kullanir - model-basina ozel
    istek govdesi gerektirmez, coklu Bedrock modeliyle (Claude, Nova, vb.)
    ayni sekilde calisir.

    BILINCLI OLARAK SENKRON (boto3'un async destegi yok, ayri bir
    aioboto3 bagimliligi gerektirir). Bu, cagrilan yerde (asyncio olay
    dongusunu) BLOKE EDER - ama photo_service._vlm_executor zaten VLM
    icin AYRILMIS, tek isci thread'inde, kendi basina bir asyncio.run()
    dongusu aciyor (bkz. Bolum 6.3.5, Mimari Rehberi); bu dongude VLM
    cagrisiyla es zamanli baska hicbir is yok, yani bloke etmenin
    baska hicbir isteği geciktirme riski yok.
    """
    client = _get_bedrock_client()
    response = client.converse(
        modelId=settings.AWS_BEDROCK_MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": PROMPT},
                    {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}},
                ],
            }
        ],
        # maxTokens: LM Studio yolundaki ayni gerekce (Bkz. _build_payload_lm_studio) -
        # dejenere/donen ciktiyi erken kesen bir ust sinir.
        inferenceConfig={"temperature": 0.1, "maxTokens": 700},
    )

    blocks = (response.get("output") or {}).get("message", {}).get("content", [])
    raw = "".join(b.get("text", "") for b in blocks if "text" in b)

    # SAGLIK KONTROLU: stopReason "end_turn" disinda bir seyse (ör.
    # "max_tokens", "content_filtered") cikti eksik/guvensiz olabilir.
    stop_reason = response.get("stopReason")
    if not raw or stop_reason not in ("end_turn", "stop_sequence"):
        raise ValueError(f"anormal vlm cevabi (stopReason={stop_reason!r}): {raw[:80]!r}")

    return raw


async def _call_vlm_bedrock(image_bytes: bytes) -> str:
    raw = _call_vlm_bedrock_sync(image_bytes)
    _check_degenerate(raw)
    return raw


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


async def _call_vlm(image_bytes: bytes) -> str:
    if IS_LM_STUDIO:
        return await _call_vlm_lm_studio(image_bytes)
    return await _call_vlm_bedrock(image_bytes)


async def analyze_photo(image_path: str) -> VLMResult:
    image_bytes = _encode_image(image_path)

    try:
        raw = await _call_vlm(image_bytes)
        result = VLMResult.model_validate_json(_clean_json(raw))
        if _is_placeholder_echo(result):
            raise ValueError("VLM sablonu aynen geri dondurdu")
        return result
    except (ValidationError, ValueError):
        # Bozuk/şablon/dejenere içerikten sonra temiz bir retry
        raw2 = await _call_vlm(image_bytes)

        result2 = VLMResult.model_validate_json(_clean_json(raw2))
        if _is_placeholder_echo(result2):
            raise ValueError("VLM sablonu tekrar ediyor, gecerli analiz uretilemedi")
        return result2
