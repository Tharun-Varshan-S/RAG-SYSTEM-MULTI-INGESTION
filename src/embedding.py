import os
import time
import logging
from typing import Optional
import google.generativeai as genai

logger = logging.getLogger(__name__)

# Attempt to configure API key from environment variable
api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyBhpF6MjzCIsIOzB3fyaiR5VgyEWxK7-uU")
if api_key:
    genai.configure(api_key=api_key)

def get_embedding(text: str, retries: int = 2) -> Optional[list[float]]:
    """
    Gets the embedding vector for the provided text using Gemini API.
    Retries up to 2 times on rate limits or API failures.
    """
    if not text or not text.strip():
        logger.error("Invalid input: empty text for embedding.")
        return None
        
    # Recommended model for text embeddings
    model = 'models/gemini-embedding-001' 
    
    for attempt in range(retries + 1):
        try:
            # Setting task_type for the document ingestion phase
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.warning(f"Embedding API failure on attempt {attempt + 1}: {str(e)}")
            if attempt < retries:
                time.sleep((attempt + 1) * 2) # Incremental backoff
            else:
                logger.error("All retry attempts failed for embedding.")
                return None
    return None
