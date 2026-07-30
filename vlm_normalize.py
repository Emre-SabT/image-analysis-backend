"""
vlm_normalize.py — PhotoAI VLM katmani icin gorsel normalizasyonu ve
guvenli Ollama cagrisi.

Teshis sonuclarindan cikan 3 duzeltmeyi uygular:
  1. EXIF rotasyonunu pikselle uygula (llama.cpp EXIF okumuyor)
  2. Uzun kenari 1024'e indir (context tasmasi + saatlerce suren istekler)
  3. JSON'u grammar seviyesinde zorla (model yorum/markdown ekleyemez)

Entegrasyon:
    from vlm_normalize import normalize_image, analyze_image
"""

import base64
import io
import json
import logging

import requests
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:3b"

# 1024 uzun kenar -> ~1100-1400 gorsel token. num_ctx=8192 icinde
# bol bol yer kalir. Kalite kaybi ihmal edilebilir, hiz kazanci buyuk.
MAX_KENAR = 1024
NUM_CTX = 8192
TIMEOUT = 120           # normalize edilmis gorsel 10 sn'de biter; 120 fazlasiyla yeter

# Ollama'ya verilecek JSON semasi. format=<schema> ile model
# baska bir sey uretemez -- markdown fence, // yorum, aciklama cumlesi yok.
SEMA = {
    "type": "object",
    "properties": {
        "caption": {"type": "string"},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "bbox_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["label"],
            },
        },
        "scene": {"type": "string"},
    },
    "required": ["caption", "objects", "scene"],
}

PROMPT = (
    "Analyze this image. Give a one-sentence caption, list the visible "
    "objects with bounding boxes, and describe the overall scene."
)


def normalize_image(path: str, max_kenar: int = MAX_KENAR) -> bytes:
    """
    Ham dosyadan modele guvenle gonderilebilir JPEG bayti uretir.

    - EXIF orientation'i piksele uygular (yan yatmis foto sorunu)
    - CMYK / RGBA / palet / grayscale -> RGB
    - Uzun kenari max_kenar'a indirir
    - Tum metadata'yi dusurur
    """
    with Image.open(path) as im:
        # EXIF rotasyonu -- teshiste ucu de orientation=6 idi, yani
        # model fotolari 90 derece yan goruyordu.
        im = ImageOps.exif_transpose(im)

        if im.mode != "RGB":
            log.info("mode donusumu: %s -> RGB (%s)", im.mode, path)
            im = im.convert("RGB")

        onceki = im.size
        if max(im.size) > max_kenar:
            im.thumbnail((max_kenar, max_kenar), Image.LANCZOS)
            log.info("kucultuldu: %sx%s -> %sx%s (%s)",
                     *onceki, *im.size, path)

        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88, optimize=True)
        return buf.getvalue()


def analyze_image(path: str, prompt: str = PROMPT) -> dict:
    """
    Gorseli normalize edip Ollama'ya gonderir, dogrulanmis dict dondurur.
    Hata durumunda RuntimeError firlatir -- cagiran katman status=failed yazar.
    """
    img_bytes = normalize_image(path)

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [base64.b64encode(img_bytes).decode()],
        "stream": False,
        "keep_alive": 0,
        "format": SEMA,                  # <-- grammar seviyesinde JSON zorlamasi
        "options": {
            "num_ctx": NUM_CTX,
            "temperature": 0,
            "repeat_penalty": 1.05,      # dejenere tekrari (@@@@) frenler
        },
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"ollama http {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    metin = (data.get("response") or "").strip()

    # SAGLIK KONTROLU: prompt_eval_count yoksa uretim normal yolla olmadi.
    # Teshiste her @@@@ vakasinda bu alan None geldi -- erken yakalama noktasi.
    if data.get("prompt_eval_count") is None:
        raise RuntimeError(
            f"anormal ollama cevabi (prompt_eval_count yok), "
            f"cikti basi: {metin[:80]!r}"
        )

    if not metin:
        raise RuntimeError("ollama bos cevap dondu")

    # Dejenere tekrar dedektoru: cikti tek karakterin tekrarindan olusuyorsa
    if len(set(metin)) <= 2 and len(metin) > 20:
        raise RuntimeError(f"dejenere cikti: {metin[:40]!r}")

    try:
        return json.loads(metin)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"json parse: {e} | cikti: {metin[:200]!r}") from e


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for p in sys.argv[1:]:
        print(f"\n=== {p} ===")
        try:
            print(json.dumps(analyze_image(p), indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"BASARISIZ -> {e}")
