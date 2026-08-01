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