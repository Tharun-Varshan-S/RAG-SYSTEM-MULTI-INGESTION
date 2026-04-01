import logging
from src.ingestion import chunk_text
from src.embedding import get_embedding
from src.retriever import add_to_database, retrieve
from src.generator import generate_response

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def ask_ai(query: str) -> dict:
    """
    Pipeline that handles validate, retrieve, generate.
    Returns the answer and sources as a dictionary.
    """
    # Validate query
    if not query or not query.strip():
        logger.warning("Empty query provided to ask_ai.")
        return {"answer": "Please provide a valid question.", "sources": []}
        
    logger.info(f"Starting query pipeline for: '{query}'")
    
    # Retrieve Embedding using the 'retrieval_query' task type conceptually 
    # (reusing the same embedding logic to keep it simple, but we can set task types similarly)
    query_emb = get_embedding(query)
    if not query_emb:
        logger.error("Failed to generate embedding for query.")
        return {"answer": "Error generating embedding for the query. Cannot proceed.", "sources": []}
        
    # Retrieve context
    retrieved_chunks = retrieve(query, query_emb, top_n=3)
    if not retrieved_chunks:
        logger.info("No relevant matches found in database.")
        return {
            "answer": "I don't have enough relevant information to answer that question.",
            "sources": []
        }
        
    # Generate Response
    answer = generate_response(query, retrieved_chunks)
    
    # Return structured output
    return {
        "answer": answer,
        "sources": retrieved_chunks
    }

def main():
    # Setup test dummy data
    raw_text = """
    Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans.
    It involves the development of algorithms and computer programs that can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.
    A vector database is a type of database that stores data as multidimensional mathematical vectors. 
    These vectors represent the features or attributes of the data in a high-dimensional space. 
    The distance between vectors can be calculated to measure similarity.
    This allows fast similarity searches rather than exact keyword matches.
    It's particularly useful for AI systems processing natural language or images.
    """
    
    print("=" * 60)
    print(" 1. DATA INGESTION & SETUP")
    print("=" * 60)
    # 1. Ingestion
    chunks = chunk_text(raw_text, chunk_size=200)
    
    # 2. Embedding + Storage
    for chunk in chunks:
        emb = get_embedding(chunk)
        if emb:
            add_to_database(chunk, emb)
            logger.info("Embeddings created for chunk.")
            
    print("\n" + "="*60)
    print(" 2. RUNNING TESTS")
    print("=" * 60)
    
    # Testing Scenarios
    tests = [
        {"desc": "Normal query", "q": "What kind of tasks require human intelligence that AI simulates?"},
        {"desc": "Keyword-heavy query", "q": "vector multidimensional mathematical distance similarity"},
        {"desc": "Vague query", "q": "It allows fast"},
        {"desc": "Empty question", "q": ""}
    ]
    
    for test in tests:
        print(f"\n[Test: {test['desc']}]")
        print(f"Query: '{test['q']}'")
        
        response = ask_ai(test['q'])
        
        print("\n--- Pipeline Result ---")
        print(f"Answer:\n{response['answer'].strip()}")
        print(f"Sources used: {len(response['sources'])}")
        for i, src in enumerate(response['sources'], 1):
            print(f"   [{i}] {src.strip()}")
        print("-" * 60)

if __name__ == "__main__":
    main()
