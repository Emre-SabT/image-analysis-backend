# ayarları tek yerden okuma

from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    DATABASE_URL = os.getenv("DATABASE_URL")
    VLM_BASE_URL = os.getenv("VLM_BASE_URL")
    VLM_MODEL = os.getenv("VLM_MODEL")
    SERVICE_KEY = os.getenv("SERVICE_KEY")
    AI_PROVIDER = os.getenv("AI_PROVIDER", "lm_studio")

settings = Settings()