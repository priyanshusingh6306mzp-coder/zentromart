# src/kb.py
import hashlib
import re

import chromadb
from sentence_transformers import SentenceTransformer

from config import INDEX_DIR, KB_DIR

COLLECTION = "zentromart_kb"
PRECEDENCE = {"pol": 1, "agent": 2, "help": 3, "faq": 4, "mkt": 5}   # lower = higher priority
SKIP_STATUS = {"superseded", "archived", "stale", "deprecated"}

_model = None
_collection = None


def _embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _front_matter(text: str):
    m = re.match(r"^\ufeff?---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    meta = {}
    if m:
        for line in m.group(1).splitlines():
            k, sep, v = line.partition(":")
            if sep:
                meta[k.strip().lower()] = v.strip().strip("\"'")
        return meta, text[m.end():]
    return meta, text


def parse_and_chunk_kb(kb_dir=KB_DIR):
    chunks, seen = [], set()
    for path in sorted(kb_dir.glob("*.md")):
        filename = path.name
        if "superseded" in filename.lower():
            continue
        meta, body = _front_matter(path.read_text(encoding="utf-8"))
        raw_sections = re.split(r"\n## ", "\n" + body)
        header = raw_sections[0]

        status = (meta.get("status")
                  or (re.search(r"Status:\s*(\w+)", header, re.I) or [None, "active"])[1]).lower()
        if status in SKIP_STATUS:
            continue
        effective = (meta.get("effective_from")
                     or (re.search(r"Effective.*?:\s*([\d-]+)", header, re.I) or [None, "unknown"])[1])
        title = (meta.get("title")
                 or (re.search(r"^#\s+(.+)$", header, re.M) or [None, filename])[1]).strip()
        doc_type = filename.split("_")[0]

        for idx, section in enumerate(raw_sections):
            if idx == 0:
                continue
            m = re.search(r"\[(.*?)\]", section)
            section_id = m.group(1) if m else f"SEC-{idx}"
            cid = f"{filename}_{section_id}"
            if cid in seen:
                cid = f"{cid}_{idx}"
            seen.add(cid)
            chunks.append({
                "id": cid,
                "text": f"Document: {title}\nSection ID: [{section_id}]\n\n{section.strip()}",
                "metadata": {"filename": filename, "section_id": section_id, "doc_type": doc_type,
                             "status": status, "effective_from": effective},
            })
    return chunks


def _fingerprint() -> str:
    h = hashlib.sha1()
    for p in sorted(KB_DIR.glob("*.md")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _get_collection():
    """Open the index; build it from KB_DIR if it is missing or the documents changed."""
    global _collection
    if _collection is not None:
        return _collection
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    fp, fp_file = _fingerprint(), INDEX_DIR / "kb_fingerprint.txt"
    col = client.get_or_create_collection(name=COLLECTION)
    stored = fp_file.read_text().strip() if fp_file.exists() else ""
    if col.count() == 0 or stored != fp:
        print(f"Building the knowledge-base index from {KB_DIR} ...", flush=True)
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
        col = client.get_or_create_collection(name=COLLECTION)
        chunks = parse_and_chunk_kb()
        if not chunks:
            raise RuntimeError(f"No knowledge-base documents found in {KB_DIR}")
        col.add(ids=[c["id"] for c in chunks],
                embeddings=_embedder().encode([c["text"] for c in chunks]).tolist(),
                documents=[c["text"] for c in chunks],
                metadatas=[c["metadata"] for c in chunks])
        fp_file.write_text(fp)
        print(f"Index ready: {len(chunks)} chunks in {INDEX_DIR}", flush=True)
    _collection = col
    return col


def build_index():
    return _get_collection()


def retrieve(query, top_k=5):
    """Retrieve chunks, drop stale ones, and re-rank by document-type precedence."""
    col = _get_collection()
    emb = _embedder().encode([query]).tolist()
    res = col.query(query_embeddings=emb, n_results=min(top_k * 2, col.count()))
    out = []
    if res["documents"] and res["documents"][0]:
        for i in range(len(res["documents"][0])):
            meta = res["metadatas"][0][i]
            if meta.get("status", "active").lower() in SKIP_STATUS:
                continue
            out.append({"id": res["ids"][0][i], "text": res["documents"][0][i], "metadata": meta,
                        "distance": res["distances"][0][i],
                        "precedence": PRECEDENCE.get(meta.get("doc_type", "mkt"), 5)})
    out.sort(key=lambda x: (x["precedence"], x["distance"]))
    return out[:top_k]


if __name__ == "__main__":
    print(f"{_get_collection().count()} chunks indexed")