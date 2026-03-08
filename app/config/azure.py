from openai import AzureOpenAI
from app.config.settings import Config

_client = None

def get_azure_client():
    global _client
    if _client is None:
        if not Config.AZURE_OPENAI_ENDPOINT or not Config.AZURE_OPENAI_API_KEY:
            raise ValueError("Azure OpenAI configuration missing. Check your .env file.")
        
        _client = AzureOpenAI(
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION
        )
    return _client
