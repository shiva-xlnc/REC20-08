import os
import re
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL_NAME, CHROMA_DIR, SEC_DATA_DIR, NTSB_DATA_DIR

embedder = SentenceTransformer(EMBED_MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name="kg_rag_corpus")


def parse_sec_filename(filename):
    """e.g. '2022 Q3 AAPL.pdf' -> company=AAPL, quarter=2022Q3"""
    match = re.match(r"(\d{4})\s*Q(\d)\s*([A-Z]+)\.pdf", filename, re.IGNORECASE)
    if match:
        year, q, company = match.groups()
        return company.upper(), f"{year}Q{q}"
    return "UNKNOWN", "UNKNOWN"


def flatten_table(table):
    """Convert a pdfplumber table (list of rows) into label:value text
    that preserves row/column meaning even out of visual context."""
    if not table or len(table) < 2:
        return None
    header = [str(h).strip() if h else "" for h in table[0]]
    lines = []
    for row in table[1:]:
        if not row or not row[0]:
            continue
        row_label = str(row[0]).strip()
        parts = []
        for col_name, val in zip(header[1:], row[1:]):
            if val is not None and str(val).strip():
                parts.append(f"{col_name}={str(val).strip()}")
        if parts:
            lines.append(f"{row_label}: " + ", ".join(parts))
    return "\n".join(lines) if lines else None


def chunk_text(text, max_words=350, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
        if start <= 0:
            break
    return chunks


def ingest_pdf(filepath, doc_type, company="UNKNOWN", quarter="UNKNOWN"):
    filename = os.path.basename(filepath)
    print(f"Ingesting {filename} ...")
    records = []

    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # --- tables first, so we don't double-count their text ---
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables):
                flat = flatten_table(table)
                if flat:
                    records.append({
                        "text": flat,
                        "chunk_type": "table",
                        "page": page_num,
                        "table_idx": t_idx,
                    })

            # --- prose text ---
            text = page.extract_text() or ""
            if text.strip():
                for c_idx, chunk in enumerate(chunk_text(text)):
                    records.append({
                        "text": chunk,
                        "chunk_type": "text",
                        "page": page_num,
                        "chunk_idx": c_idx,
                    })

    # embed + store
    ids, docs, metadatas, embeddings = [], [], [], []
    for i, rec in enumerate(records):
        chunk_id = f"{filename}_p{rec['page']}_{rec['chunk_type']}_{i}"
        ids.append(chunk_id)
        docs.append(rec["text"])
        metadatas.append({
            "source_doc": filename,
            "doc_type": doc_type,
            "company": company,
            "quarter": quarter,
            "page": rec["page"],
            "chunk_type": rec["chunk_type"],
        })

    if not docs:
        print(f"  no extractable content in {filename}")
        return

    embeddings = embedder.encode(docs, show_progress_bar=False).tolist()
    collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
    print(f"  added {len(docs)} chunks ({sum(1 for r in records if r['chunk_type']=='table')} table, "
          f"{sum(1 for r in records if r['chunk_type']=='text')} text)")


def ingest_all():
    # SEC 10-Q
    if os.path.exists(SEC_DATA_DIR):
        for filename in os.listdir(SEC_DATA_DIR):
            if filename.lower().endswith(".pdf"):
                company, quarter = parse_sec_filename(filename)
                ingest_pdf(os.path.join(SEC_DATA_DIR, filename), "sec10q", company, quarter)

    # NTSB
    if os.path.exists(NTSB_DATA_DIR):
        for filename in os.listdir(NTSB_DATA_DIR):
            if filename.lower().endswith(".pdf"):
                ingest_pdf(os.path.join(NTSB_DATA_DIR, filename), "ntsb")

    print(f"\nTotal chunks in collection: {collection.count()}")


if __name__ == "__main__":
    ingest_all()