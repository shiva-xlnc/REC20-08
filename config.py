import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"
SEC_DATA_DIR = "./data/sec10q"
NTSB_DATA_DIR = "./data/ntsb"

# top-k for retrieval
TOP_K_SINGLE = 5
TOP_K_MULTI = 10

# known SEC tickers in this dataset — used for entity detection
KNOWN_COMPANIES = ["AAPL", "AMZN", "INTC", "MSFT", "NVDA"]