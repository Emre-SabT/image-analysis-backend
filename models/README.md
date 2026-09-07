# models/

Bu klasörün içeriği `.gitignore`'dadır (bu dosya hariç) — ONNX/safetensors
ağırlıkları yüzlerce MB, repoyu şişirir. Backend'i çalıştırmadan önce
aşağıdaki üç model dosyasını **elle indirmeniz** gerekir. Toplam: **~1,4 GB**.

Tam kurulum rehberi için bkz. [`../KURULUM.md`](../KURULUM.md) — Adım 6.

```bash
mkdir -p models/auraface
```

## 1. YuNet — Yüz Tespiti

| | |
|---|---|
| Hedef yol | `models/face_detection_yunet_2023mar.onnx` |
| Boyut | ~233 KB |
| Ayar | `YUNET_MODEL_PATH` (`.env`) |
| Lisans | MIT |
| Kaynak | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet |
| Kullanan kod | `app/ai/face_detector.py` |

```bash
curl -L -o models/face_detection_yunet_2023mar.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
```

Doğrulama: `(Get-Item models\face_detection_yunet_2023mar.onnx).Length` → **232589** civarı.
1 KB civarındaysa Git LFS işaretçisi indirmişsinizdir — `-L` bayrağını unutmayın.

## 2. AuraFace v1 — Yüz Embedding

| | |
|---|---|
| Hedef yol | `models/auraface/glintr100.onnx` |
| Boyut | ~249 MiB |
| Ayar | `AURAFACE_MODEL_DIR` (`.env`) |
| Lisans | Apache-2.0 |
| Kaynak | https://huggingface.co/fal/AuraFace-v1 |
| Kullanan kod | `app/ai/face_embedder.py` |

```bash
hf download fal/AuraFace-v1 glintr100.onnx --local-dir models/auraface
```

Depoda `scrfd_10g_bnkps.onnx`, `genderage.onnx` gibi başka dosyalar da vardır —
**yalnızca `glintr100.onnx` gereklidir** (yüz tespiti YuNet ile yapılıyor,
AuraFace'in kendi SCRFD dedektörü kullanılmıyor).

## 3. multilingual-e5-base — Semantik Arama Embedding'i

| | |
|---|---|
| Hedef yol | `models/multilingual-e5-base/` (klasörün tamamı) |
| Boyut | ~1,08 GiB |
| Ayar | `EMBEDDING_MODEL_DIR`, `EMBEDDING_DIM=768` (`.env`) |
| Lisans | MIT |
| Kaynak | https://huggingface.co/intfloat/multilingual-e5-base |
| Kullanan kod | `app/ai/embedding/local_e5.py` |

```bash
hf download intfloat/multilingual-e5-base --local-dir models/multilingual-e5-base
```

Şu dosyalar mutlaka bulunmalı: `config.json`, `model.safetensors`, `modules.json`,
`sentence_bert_config.json`, `sentencepiece.bpe.model`, `special_tokens_map.json`,
`tokenizer.json`, `tokenizer_config.json`, `1_Pooling/config.json`.

> `SEMANTIC_SEARCH_ENABLED=false` yapılırsa bu model gerekmez.

---

`hf` komutu `huggingface_hub` paketinin CLI'ıdır (`requirements.txt` ile gelir).
Alternatif indirme yöntemi ve hata çözümleri için [`../KURULUM.md`](../KURULUM.md)
dosyasındaki Adım 6 ve "Yaygın Hatalar" (K3, K4, K8) bölümlerine bakın.
