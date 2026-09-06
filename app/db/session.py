# veritabanı bağlantısı

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.settings import settings

# HAVUZ BOYUTU BILINCLI OLARAK AYARLI (varsayilan DEGIL).
#
# SQLAlchemy varsayilani pool_size=5 + max_overflow=10 = 15 baglanti
# tavanidir. Frontend yukleme penceresi sabit 3'ten adaptif 2-12'ye
# cikarildiginda (bkz. photoai-frontend features/upload/constants.ts) bu
# tavan yetmiyor: es zamanli 12 POST /photos + durum yoklamasi
# (GET /photos/status) + liste + auth ayni havuzu paylasir ve asildiginda
# SQLAlchemy "QueuePool limit ... timeout" firlatir - kullanici tarafinda
# rastgele 500'ler olarak gorunur.
#
# 10 + 15 = 25 tavan: API sureci icin rahat, ama Postgres'in varsayilan
# max_connections=100'unu ZORLAMAZ - ayni engine'i worker surecleri de
# import eder (her surec KENDI havuzunu acar). Worker'lar is basina kisa
# omurlu session kullandigi ve surec basina tek is islediginden pratikte
# 2-3 baglantinin uzerine cikmazlar; tavan yine de yatay olceklemede
# guvenli kalsin diye olculu tutuldu.
#
# pool_pre_ping: bosta kalmis baglanti sunucu/proxy tarafindan kapatilmis
# olabilir (uzun suren backfill'lerden sonra tipik) - kullanmadan once
# dogrulanir, boylece ilk istek "server closed the connection" ile patlamaz.
# pool_recycle: baglantilar 30 dk sonra tazelenir (ayni sorunun onlemi).
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=15,
    pool_pre_ping=True,
    pool_recycle=1800,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
