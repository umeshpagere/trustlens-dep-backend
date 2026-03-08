import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    PORT = int(os.getenv("PORT", 5000))

    # Google Fact Check Tools API (Phase 2)
    GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")

    # Deepgram Speech-to-Text (Tier 2 Video Analysis)
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

    # News API (evidence aggregation)
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")

    # Azure Computer Vision (video frame OCR)
    AZURE_VISION_KEY = os.getenv("AZURE_VISION_KEY")
    AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")

    # Sightengine (video AI-generated frame detection)
    SIGHTENGINE_API_USER = os.getenv("SIGHTENGINE_API_USER")
    SIGHTENGINE_API_SECRET = os.getenv("SIGHTENGINE_API_SECRET")

    # MongoDB configuration
    MONGODB_URI = os.getenv("MONGODB_URI")
    MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "trustlensDB")
    MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "analysis_records")
    MONGODB_TLS_ALLOW_INVALID_CERTIFICATES = os.getenv("MONGODB_TLS_ALLOW_INVALID_CERTIFICATES", "false").lower() == "true"
