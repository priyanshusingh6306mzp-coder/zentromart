# src/kb.py
import os
import re
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Initialize local embedding model and DB
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="zentromart_kb")

# Define precedence mapping (Lower number = higher priority)
PRECEDENCE = {"pol": 1, "agent": 2, "help": 3, "faq": 4, "mkt": 5}

def parse_and_chunk_kb(kb_dir):
    chunks = []
    for filename in os.listdir(kb_dir):
        if not filename.endswith(".md"):
            continue
            
        # CRITICAL RULE: Exclude superseded docs
        if "superseded" in filename.lower():
            continue
            
        filepath = os.path.join(kb_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        doc_type = filename.split('_')[0] 
        raw_sections = re.split(r'\n## ', content)
        doc_title = raw_sections[0].split('\n')[0].replace('# ', '').strip()
        
        for idx, section in enumerate(raw_sections):
            if idx == 0:
                continue 
                
            section_id_match = re.search(r'\[(.*?)\]', section)
            section_id = section_id_match.group(1) if section_id_match else f"SEC-{idx}"
            status_match = re.search(r'Status:\s*(\w+)', section, re.IGNORECASE)
            date_match = re.search(r'Effective.*:\s*([\d-]+)', section, re.IGNORECASE)
            
            chunk_text = f"Document: {doc_title}\nSection ID: [{section_id}]\n\n{section.strip()}"
            
            chunks.append({
                "id": f"{filename}_{section_id}",
                "text": chunk_text,
                "metadata": {
                    "filename": filename,
                    "section_id": section_id,
                    "doc_type": doc_type,
                    "status": status_match.group(1) if status_match else "active",
                    "effective_from": date_match.group(1) if date_match else "unknown"
                }
            })
    return chunks

def build_index():
    kb_path = os.path.join(os.path.dirname(__file__), '..', 'kb')
    chunks = parse_and_chunk_kb(kb_path)
    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    embeddings = embedding_model.encode(documents).tolist()
    
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"Indexing complete! Saved to {DB_PATH}")

def retrieve(query, top_k=5):
    """Retrieves chunks, filters stale ones, and re-ranks by doc type precedence."""
    query_embedding = embedding_model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k * 2 
    )
    
    retrieved_chunks = []
    if results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            meta = results['metadatas'][0][i]
            
            status = meta.get('status', 'active').lower()
            if status in ['superseded', 'archived', 'stale']:
                continue
                
            retrieved_chunks.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": meta,
                "distance": results['distances'][0][i],
                "precedence": PRECEDENCE.get(meta.get('doc_type', 'mkt'), 5)
            })
            
    retrieved_chunks.sort(key=lambda x: (x['precedence'], x['distance']))
    return retrieved_chunks[:top_k]

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        build_index()