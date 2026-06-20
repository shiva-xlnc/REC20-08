import re
from sentence_transformers import SentenceTransformer
import chromadb
from config import EMBED_MODEL_NAME, CHROMA_DIR, TOP_K_SINGLE, TOP_K_MULTI, KNOWN_COMPANIES

embedder = SentenceTransformer(EMBED_MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name="kg_rag_corpus")


def detect_companies(question):
    found = [c for c in KNOWN_COMPANIES if re.search(rf"\b{c}\b", question, re.IGNORECASE)]
    return found


def vector_search(question, k=TOP_K_SINGLE, where_filter=None):
    query_emb = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_emb,
        n_results=k,
        where=where_filter if where_filter else None,
    )
    chunks = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        chunks.append({"text": doc, "metadata": meta, "score": dist})
    return chunks


def retrieve(question):
    """Tiered router:
    - Tier 3: 2+ companies mentioned -> retrieve separately per company
    - Tier 1/2: single-entity or general -> standard top-k (k scales if question looks broad)
    """
    companies = detect_companies(question)

    if len(companies) >= 2:
        # Tier 3: cross-document
        all_chunks = []
        for company in companies:
            chunks = vector_search(question, k=6, where_filter={"company": company})
            all_chunks.extend(chunks)
        return all_chunks, "cross_document"

    elif len(companies) == 1:
        # Tier 1/2: filter to that company, but allow more chunks
        # in case the question needs multiple parts of the same doc (table + text)
        broad_keywords = ["summarize", "summary", "overall", "explain", "describe", "trend"]
        k = TOP_K_MULTI if any(w in question.lower() for w in broad_keywords) else TOP_K_SINGLE
        chunks = vector_search(question, k=k, where_filter={"company": companies[0]})
        tier = "multi_chunk_same_doc" if k == TOP_K_MULTI else "single_chunk"
        return chunks, tier

    else:
        # No company detected (e.g. NTSB questions, or generic) -> plain top-k
        chunks = vector_search(question, k=TOP_K_SINGLE)
        return chunks, "single_chunk"