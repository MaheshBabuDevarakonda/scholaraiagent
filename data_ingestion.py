import os
import json
import math
from typing import List, Dict, Any

# Simple local vector store database path
INDEX_PATH = "vector_index.json"

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error loading Gemini GenAI SDK: {e}")
        return None

# Chunking logic: sliding window / section-based chunker
def chunk_document(filepath: str) -> List[str]:
    chunks = []
    if not os.path.exists(filepath):
        return chunks
        
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    current_region = "Global"
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Detect section header
        if line.startswith("=== REGION:"):
            current_region = line.replace("===", "").replace("REGION:", "").strip()
            continue
        
        # Append region to maintain context during search
        if line.startswith("-"):
            chunk_text = f"[{current_region}] {line[1:].strip()}"
            chunks.append(chunk_text)
            
    return chunks

# Fallback TF-IDF style string intersection for mock embeddings if offline/no key
def generate_mock_embedding(text: str) -> List[float]:
    # Produce a simple deterministic 128-dimensional vector based on word hashes
    dim = 128
    vector = [0.0] * dim
    words = text.lower().replace("[", "").replace("]", "").replace("-", "").split()
    for w in words:
        # Simple hashing to spread weights across dimensions
        h = hash(w) % dim
        vector[h] += 1.0
        
    # L2 Normalization
    magnitude = math.sqrt(sum(x * x for x in vector))
    if magnitude > 0:
        vector = [x / magnitude for x in vector]
    return vector

# Compute Cosine Similarity between two vectors
def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if len(v1) != len(v2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 * mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)

# Generate embeddings and save to vector index
def build_vector_index(filepath: str = "knowledge_base.txt"):
    chunks = chunk_document(filepath)
    if not chunks:
        print("No chunks found in document.")
        return

    print(f"Ingested document into {len(chunks)} chunks.")
    
    client = get_gemini_client()
    index_data = []
    
    if client:
        print("GEMINI_API_KEY detected. Generating embeddings via Gemini text-embedding-004...")
        try:
            for idx, chunk in enumerate(chunks):
                # Call Gemini Embedding API
                response = client.models.embed_content(
                    model="text-embedding-004",
                    contents=chunk
                )
                embedding_vector = response.embeddings[0].values
                index_data.append({
                    "id": idx,
                    "chunk_text": chunk,
                    "embedding": embedding_vector,
                    "is_mock": False
                })
        except Exception as e:
            print(f"Error calling Gemini Embedding API: {e}. Falling back to mock embeddings.")
            client = None  # Force mock fallback
            
    if not client:
        print("Running in Mock Mode. Generating local rule-based text embeddings...")
        for idx, chunk in enumerate(chunks):
            embedding_vector = generate_mock_embedding(chunk)
            index_data.append({
                "id": idx,
                "chunk_text": chunk,
                "embedding": embedding_vector,
                "is_mock": True
            })
            
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)
        
    print(f"Successfully saved vector database index to {INDEX_PATH}")

# Search vector index for the top K matching chunks
def query_vector_index(query: str, k: int = 3) -> List[Dict[str, Any]]:
    if not os.path.exists(INDEX_PATH):
        # Build index on the fly if missing
        build_vector_index()
        
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index_data = json.load(f)
        
    if not index_data:
        return []
        
    # Check if index was built in mock mode
    index_is_mock = index_data[0].get("is_mock", False)
    client = get_gemini_client()
    
    query_vector = None
    if client and not index_is_mock:
        try:
            response = client.models.embed_content(
                model="text-embedding-004",
                contents=query
            )
            query_vector = response.embeddings[0].values
        except Exception as e:
            print(f"Error fetching query embedding: {e}. Falling back to mock search.")
            query_vector = None
            
    if not query_vector:
        # Fallback to mock query embedding
        query_vector = generate_mock_embedding(query)
        
    # Score all chunks using Cosine Similarity + Region & Purpose Keyword Boosting
    scored_chunks = []
    query_lower = query.lower()
    
    # Detect region targets in query
    boost_region = None
    if "uk" in query_lower or "united kingdom" in query_lower:
        boost_region = "UNITED KINGDOM"
    elif "canada" in query_lower:
        boost_region = "CANADA"
    elif "australia" in query_lower:
        boost_region = "AUSTRALIA"
    elif "usa" in query_lower or "united states" in query_lower:
        boost_region = "UNITED STATES"
        
    # Detect purpose targets in query
    boost_purpose = None
    if "study" in query_lower or "student" in query_lower:
        boost_purpose = "study"
    elif "work" in query_lower or "job" in query_lower or "skilled" in query_lower:
        boost_purpose = "work"
    elif "tourist" in query_lower or "visitor" in query_lower:
        boost_purpose = "visitor"
    elif "pr" in query_lower or "permanent" in query_lower or "ilr" in query_lower:
        boost_purpose = "pr"
        
    for item in index_data:
        sim = cosine_similarity(query_vector, item["embedding"])
        
        # Apply metadata boosts
        chunk_text = item["chunk_text"]
        chunk_text_lower = chunk_text.lower()
        
        # 1. Region Boosting (+0.35)
        if boost_region:
            if f"[{boost_region}" in chunk_text.upper() or boost_region in chunk_text.upper():
                sim += 0.35
                
        # 2. Purpose Boosting (+0.25)
        if boost_purpose:
            if boost_purpose == "study" and "study" in chunk_text_lower:
                sim += 0.25
            elif boost_purpose == "work" and ("work" in chunk_text_lower or "skilled worker" in chunk_text_lower):
                sim += 0.25
            elif boost_purpose == "visitor" and ("tourist" in chunk_text_lower or "visitor" in chunk_text_lower):
                sim += 0.25
            elif boost_purpose == "pr" and ("pr" in chunk_text_lower or "indefinite" in chunk_text_lower):
                sim += 0.25
                
        scored_chunks.append({
            "chunk_text": chunk_text,
            "similarity": sim
        })
        
    # Sort by similarity descending
    scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
    return scored_chunks[:k]

if __name__ == "__main__":
    build_vector_index()

