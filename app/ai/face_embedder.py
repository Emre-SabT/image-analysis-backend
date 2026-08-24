"""
AuraFace tabanli yuz embedding modulu.

PhotoAI AI katmaninin Bilesen 2'si: Yuz Embeddingi.
Ticari-uyumlu alternatif pipeline karari geregi ArcFace yerine AuraFace
kullaniliyor (bkz. PhotoAI_Alternatif_Pipeline.docx, Bolum 3.3 - Neden AuraFace?).

AuraFace'in kendi bundled SCRFD detektoru KULLANILMIYOR - detection icin
Faz 1'de dogrulanan YuNet (face_detector.py) kullanilmaya devam ediliyor.
Sadece recognition/embedding modeli (glintr100.onnx) buradan aliniyor.

Colab'da (Faz 2, 30 fotografli 5-kisilik test setinde) dogrulandi:
ayni-kisi ort. benzerlik 0.581, farkli-kisi ort. benzerlik 0.088,
aralarinda ORTUSME YOK (temiz ayrim). Detaylar icin staj gunlugune bak.
"""

import os
from pathlib import Path

import numpy as np
from insightface.model_zoo import model_zoo
from insightface.utils import face_align


class FaceEmbedder:
    """
    AuraFace (glintr100.onnx) embedding modelinin ince bir sarmalayicisi.

    Kullanim:
        embedder = FaceEmbedder(model_dir="models/auraface")
        embedding, aligned_crop = embedder.get_embedding(image_bgr, detected_face.landmarks)

    Not - CPU tercihi: VLM (Qwen2.5-VL-7B) mevcut donanimda VRAM'in buyuk
    kismini kullaniyor (bkz. staj gunlugu, VRAM debug'i). AuraFace de YuNet
    gibi bilincli olarak CPU'da calisiyor - GPU'ya hic dokunmuyor, VLM'le
    catisma olmuyor.
    """

    EMBEDDING_DIM = 512

    def __init__(self, model_dir: str = "models/auraface"):
        model_path = Path(model_dir) / "glintr100.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"AuraFace embedding modeli bulunamadi: {model_path}\n"
                "Indirme: huggingface_hub.snapshot_download('fal/AuraFace-v1', "
                f"local_dir='{model_dir}')"
            )

        self._model = model_zoo.get_model(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._model.prepare(ctx_id=-1)  # -1 = CPU

    def get_embedding(
        self,
        image_bgr: np.ndarray,
        landmarks: list[tuple[float, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        5 landmark noktasiyla yuzu 112x112 hizalar, 512-d L2-normalize
        embedding uretir.

        image_bgr: cv2.imread(...) ile okunmus BGR goruntu (YuNet'e verilenle
            AYNI olcekte olmali - resize_for_detection sonrasi goruntu).
        landmarks: FaceDetector.detect()'ten gelen DetectedFace.landmarks
            (sag goz, sol goz, burun, sag agiz, sol agiz sirasiyla).

        Donus: (embedding, aligned_crop)
            embedding: (512,) sekilli, normu 1.0 olan float32 numpy array.
            aligned_crop: (112, 112, 3) hizalanmis BGR yuz kirpimi - MinIO'ya
                face_crop_key olarak kaydedilecek gorsel (Bolum 11.1, faces tablosu).

        Onemli - landmark sirasi: Eger hizalanmis kirpimlar carpik/aynali
        cikiyorsa (gorsel kontrolde fark edilir), su satiri ekle:
            landmark_arr = landmark_arr[[1, 0, 2, 4, 3]]
        Bu, Faz 2'de karsilasilan ve gorsel kontrolle dogrulanan bir risk
        noktasiydi; test setinde sorun cikmadi ama farkli kaynakli
        goruntulerde (ör. farkli kamera/on-isleme) tekrar kontrol edilmeli.
        """
        landmark_arr = np.array(landmarks, dtype=np.float32)
        aligned = face_align.norm_crop(image_bgr, landmark_arr, image_size=112)

        feat = np.asarray(self._model.get_feat(aligned)).reshape(-1)
        feat = feat / np.linalg.norm(feat)
        return feat, aligned


# --- Kimlik atama esigi (Bolum 8.2'nin AuraFace icin yeniden kalibre edilmis
#     hali - orijinal 0.55/0.62/0.70 ArcFace icindi, AuraFace farkli bir
#     olcekte calisiyor) ---
#
# ONEMLI: Bu deger kucuk bir POC setinden (30 foto, 5 kisi) turetildi, ama
# artik iki bagimsiz canli veri setinde (180 ve 718 yuz, leave-one-out
# simulasyonuyla) dogrulandi - yukseltilmemesi gerektigi net (bkz.
# scripts/test_hybrid_assignment.py sonuclari).
#
# NOT (temizlik gecmisi): SUGGEST_THRESHOLD (0.35) ve SAME_PERSON_THRESHOLD
# (0.40), hedeflenen tasarimin 3 kademeli (SUGGEST/benzer-kisi/OTOMATIK)
# esik bandinin kalintisiydi - kullanicinin acik istegiyle tek esige
# indirgendiginde kullanimdan kalkmislardi (hicbir aktif kod yolunda
# okunmuyorlardi), simdi kaldirildilar.
AUTO_ASSIGN_THRESHOLD = 0.55  # Bu skorun ustunde: otomatik kimlik atamasi (FR-07)
