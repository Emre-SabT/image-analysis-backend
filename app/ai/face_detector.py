"""
YuNet tabanli yuz tespit modulu.

PhotoAI AI katmaninin Bilesen 1'i: Yuz Tespiti.
Ticari-uyumlu alternatif pipeline karari geregi SCRFD yerine YuNet kullaniliyor
(bkz. PhotoAI_Alternatif_Pipeline.docx, Bolum 3.2 - Neden YuNet?).

YuNet, OpenCV'nin cv2.FaceDetectorYN sinifi uzerinden calisir; ayri bir
kutuphane kurulumu gerekmez, sadece opencv-python (>=4.5.4) yeterlidir.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DetectedFace:
    """Tek bir tespit edilen yuzun verisi."""

    bbox: tuple[int, int, int, int]        # x, y, w, h (piksel)
    landmarks: list[tuple[float, float]]   # 5 nokta: sag goz, sol goz, burun, sag agiz, sol agiz
    confidence: float


class FaceDetector:
    """
    YuNet yuz dedektorunun ince bir sarmalayicisi.

    Kullanim:
        detector = FaceDetector(model_path="models/face_detection_yunet_2023mar.onnx")
        faces = detector.detect(image_bgr)

    Not: AuraFace/ArcFace hizalamasi icin ihtiyac duyulan 5-nokta landmark
    formatiyla birebir uyumludur (bkz. dokuman Bolum 3.2 - "Uyum" maddesi),
    yani Faz 2'de embedder'a gecerken hizalama adimi degismeden calisacak.

    Onemli - GPU/VRAM notu: VLM (Qwen2.5-VL-7B) mevcut 6 GB VRAM'in ~5.75 GB'ini
    kullaniyor (bkz. staj gunlugu, VRAM yetersizligi debug'i). YuNet zaten cok
    hafif oldugu icin (bu, secim gerekcelerinden biriydi) burada backend/target
    bilincli olarak CPU'ya sabitlendi - GPU'ya hic dokunmaz, VLM'le catisma
    olmaz. AuraFace (Faz 2) icin de ayni strateji uygulanacak.
    """

    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ):
        model_path = str(Path(model_path))
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"YuNet model dosyasi bulunamadi: {model_path}\n"
                "Indirme adresi: https://github.com/opencv/opencv_zoo/raw/main/"
                "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            )

        # input_size burada placeholder; her detect() cagrisinda gercek
        # goruntu boyutuna gore guncelleniyor (asagida setInputSize).
        #
        # backend_id/target_id bilincli olarak CPU'ya sabitlendi
        # (DNN_BACKEND_OPENCV / DNN_TARGET_CPU): VLM'in GPU'yu doldurdugu
        # bir ortamda YuNet'in yanlislikla CUDA'ya kaymasini engeller.
        self._detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """
        Verilen BGR goruntude yuzleri tespit eder.

        image_bgr: cv2.imread(...) ile okunmus, BGR formatinda numpy array.
        """
        h, w = image_bgr.shape[:2]
        self._detector.setInputSize((w, h))

        _, raw_faces = self._detector.detect(image_bgr)
        if raw_faces is None:
            return []

        results: list[DetectedFace] = []
        for row in raw_faces:
            x, y, fw, fh = row[0:4].astype(int)
            landmarks = [
                (float(row[4]), float(row[5])),    # sag goz
                (float(row[6]), float(row[7])),    # sol goz
                (float(row[8]), float(row[9])),    # burun
                (float(row[10]), float(row[11])),  # sag agiz kosesi
                (float(row[12]), float(row[13])),  # sol agiz kosesi
            ]
            confidence = float(row[14])
            results.append(
                DetectedFace(
                    bbox=(int(x), int(y), int(fw), int(fh)),
                    landmarks=landmarks,
                    confidence=confidence,
                )
            )
        return results
