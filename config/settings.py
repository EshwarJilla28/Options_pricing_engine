import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Database
    MONGODB_URL = "mongodb://localhost:27017/"
    DATABASE_NAME = "options_pricing"
    
    # API Settings
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
    FMP_API_KEY = os.getenv("FMP_API_KEY", "")
    MARKETDATA_TOKEN = os.getenv("MARKETDATA_TOKEN", "")
    
    # Cache Settings
    CACHE_TIMEOUT = 900  # 15 minutes
    
    # Default Parameters
    DEFAULT_RISK_FREE_RATE = 0.05
    DEFAULT_VOLATILITY = 0.25
    
    # Supported Stocks
    SUPPORTED_STOCKS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
        "META", "NVDA", "JPM", "JNJ", "V"
    ]
    
    # API Rate Limits (calls per day for free tiers)
    API_LIMITS = {
        "ALPHA_VANTAGE": 25,
        "POLYGON": 5,
        "FMP": 250
    }

settings = Settings()
