"""Worker'in JOB_TYPES fail-fast dogrulamasi.

Neden kritik: varsayilan "tum tipleri tuket" davranisi OLSAYDI, yanlis
yapilandirilmis bir worker-face konteyneri sessizce VLM islerini de
cekmeye baslardi ve tek GPU'yu paylasan VLM icin replica-bazli
eszamanlilik kontrolu (kaldirilan Semaphore(1)'in yerini alan mekanizma)
anlamsiz hale gelirdi.
"""

import pytest

from app.db.models import JOB_TYPE_FACE_PIPELINE, JOB_TYPE_VLM_ANALYSIS
from app.worker.main import parse_job_types


@pytest.mark.parametrize("raw", [None, "", "   ", ",", " , , "])
def test_missing_or_empty_job_types_refuses_to_start(raw):
    with pytest.raises(SystemExit) as exc:
        parse_job_types(raw)
    assert exc.value.code != 0, "Gecersiz JOB_TYPES ile exit code SIFIR OLMAMALI"


def test_unknown_job_type_refuses_to_start():
    with pytest.raises(SystemExit) as exc:
        parse_job_types("vlm_analysis,uydurma_tip")
    assert "uydurma_tip" in str(exc.value)


def test_valid_single_type():
    assert parse_job_types("vlm_analysis") == [JOB_TYPE_VLM_ANALYSIS]


def test_valid_single_type_with_whitespace():
    assert parse_job_types("  face_pipeline  ") == [JOB_TYPE_FACE_PIPELINE]


def test_multiple_types_allowed_but_warned(caplog):
    """Coklu tip CALISIR ama uyari verir (yavas ANY yoluna duser)."""
    import logging

    with caplog.at_level(logging.WARNING):
        result = parse_job_types("vlm_analysis,face_pipeline")

    assert result == [JOB_TYPE_VLM_ANALYSIS, JOB_TYPE_FACE_PIPELINE]
    assert any("YAVAS" in r.message or "yavas" in r.message.lower()
               for r in caplog.records), "Coklu tipte uyari verilmeli"
