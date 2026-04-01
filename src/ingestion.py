import logging

logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 250) -> list[str]:
    """
    Chunks raw text into smaller segments of roughly chunk_size characters.
    Handles empty, short, and large text.
    """
    if not text or not text.strip():
        logger.warning("Empty input text provided for chunking.")
        return []
        
    text = text.strip()
    if len(text) <= chunk_size:
        logger.info("Text is shorter than chunk size, returning as single chunk.")
        return [text]
        
    # Simple chunking by length for demonstration
    # In production, sentence boundary chunking is better recommended
    chunks = []
    
    # Using a simple sliding window strategy
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size].strip()
        if chunk:
            chunks.append(chunk)
            
    logger.info(f"Chunking done: {len(chunks)} chunks created out of {len(text)} characters.")
    return chunks
