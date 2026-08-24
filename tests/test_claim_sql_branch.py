"""claim_next'in tek/coklu job_type dallanmasinin REGRESYON testi.

Neden bu test var:
  Adim 5 olcumunde `type = ANY(:job_types)` ile Postgres'in indeksten gelen
  siralamayi kullanamadigi (ScalarArrayOpExpr -> zorunlu `Sort` dugumu) ve
  duz esitlikte ~266x hizlandigi olculdu (200.000 satirda 9,582 ms -> 0,036 ms).

  Bu optimizasyon SESSIZCE bozulabilir: ileride biri JOB_TYPES'i coklu tipe
  genisletirse (ornegin JOB_TYPES=vlm_analysis,face_pipeline) sistem HATASIZ
  calismaya devam eder ama fark edilmeden yavas yola duser. Bu test o
  regresyonu yakalar.

Testler TEMP bir `jobs` tablosuna karsi calisir (bkz. conftest.jobs_conn),
canli veriye dokunmaz ve migration uygulanmis olsun ya da olmasin gecer.
"""

from sqlalchemy import text

from app.db import jobs_repository as jr


# --- 1) Uretilen SQL metni (DB gerektirmez) ----------------------------


def test_single_type_uses_plain_equality():
    sql = jr.build_claim_sql(["vlm_analysis"])
    assert "type = :job_type" in sql, "Tek tipte DUZ ESITLIK kullanilmali"
    assert "ANY(" not in sql, "Tek tipte ANY(...) KULLANILMAMALI (Sort'a yol acar)"


def test_single_type_params_bind_scalar():
    params = jr.build_claim_params("w1", ["vlm_analysis"])
    assert params["job_type"] == "vlm_analysis"
    assert "job_types" not in params, "Tek tipte dizi parametresi baglanmamali"


def test_multi_type_falls_back_to_any():
    sql = jr.build_claim_sql(["vlm_analysis", "face_pipeline"])
    assert "type = ANY(:job_types)" in sql, "Coklu tipte ANY yedek yolu kullanilmali"
    assert "type = :job_type" not in sql


def test_multi_type_params_bind_list():
    params = jr.build_claim_params("w1", ["vlm_analysis", "face_pipeline"])
    assert params["job_types"] == ["vlm_analysis", "face_pipeline"]
    assert "job_type" not in params


# --- 2) Gercek sorgu plani (EXPLAIN) -----------------------------------


def _explain(conn, job_types):
    sql = jr.build_claim_sql(job_types)
    params = jr.build_claim_params("explain-worker", job_types)
    rows = conn.execute(
        text("EXPLAIN (ANALYZE, BUFFERS) " + sql), params
    ).fetchall()
    return "\n".join(r[0] for r in rows)


def test_single_type_plan_has_no_sort_and_uses_claim_index(seeded_jobs_conn):
    """Tek tip: Sort dugumu OLMAMALI, claim indeksi kullanilmali, Seq Scan olmamali."""
    plan = _explain(seeded_jobs_conn, ["vlm_analysis"])
    seeded_jobs_conn.rollback()  # EXPLAIN ANALYZE gercekten UPDATE eder - geri al

    assert "Sort" not in plan, f"Tek tipte Sort dugumu OLMAMALI. Plan:\n{plan}"
    assert "ix_tmp_jobs_claim" in plan, f"claim indeksi kullanilmali. Plan:\n{plan}"
    assert "Seq Scan" not in plan, f"Seq Scan olmamali. Plan:\n{plan}"


def test_multi_type_path_actually_works(seeded_jobs_conn):
    """Coklu tip yolu CALISMALI (yavas olmasi beklenen davranis, sorun degil)."""
    plan = _explain(seeded_jobs_conn, ["vlm_analysis", "face_pipeline"])
    seeded_jobs_conn.rollback()

    # Yol calisiyor ve gercekten ANY predikatini kullaniyor.
    assert "ANY" in plan, f"Coklu tip yolunda ANY predikati gorunmeli. Plan:\n{plan}"


def test_multi_type_is_the_slower_path(seeded_jobs_conn):
    """Coklu tipin Sort'a dustugunu ACIKCA belgele.

    Bu bir 'basarisizlik' degil - beklenen ve kabul edilen davranis.
    Test, iki yol arasindaki farkin GERCEK oldugunu ve tek tip yolunun
    neden korunmasi gerektigini kanitlar.
    """
    single = _explain(seeded_jobs_conn, ["vlm_analysis"])
    seeded_jobs_conn.rollback()
    multi = _explain(seeded_jobs_conn, ["vlm_analysis", "face_pipeline"])
    seeded_jobs_conn.rollback()

    assert "Sort" not in single
    assert "Sort" in multi, (
        "Coklu tip yolunun Sort'a dustugu varsayimi artik gecerli degil - "
        "Postgres davranisi degismis olabilir, dallanma yeniden olculmeli.\n"
        f"Plan:\n{multi}"
    )
