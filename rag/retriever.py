import os
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"
CHROMA_DB_PATH = Path(__file__).parent.parent / "memory" / "chroma_db"
COLLECTION_NAME = "math_knowledge"
MEMORY_COLLECTION = "solved_problems"


def get_embedding_function():
    """Use BAAI/bge-small-en-v1.5 via sentence-transformers."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )


def get_chroma_client():
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DB_PATH))


def ingest_knowledge_base():
    """Chunk and embed all knowledge base docs into ChromaDB."""
    client = get_chroma_client()
    ef = get_embedding_function()

    # Delete existing collection to re-ingest
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    docs, ids, metadatas = [], [], []
    chunk_id = 0

    for txt_file in KNOWLEDGE_BASE_DIR.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8")
        topic = txt_file.stem

        # Chunk by double newline (paragraph-level)
        chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 40]

        for chunk in chunks:
            docs.append(chunk)
            ids.append(f"{topic}_{chunk_id}")
            metadatas.append({"source": txt_file.name, "topic": topic})
            chunk_id += 1

    if docs:
        collection.add(documents=docs, ids=ids, metadatas=metadatas)
        logger.info(f"Ingested {len(docs)} chunks from knowledge base.")
    return len(docs)


def retrieve_relevant_chunks(query: str, top_k: int = 4) -> list[dict]:
    """Retrieve top-k relevant chunks for a query."""
    client = get_chroma_client()
    ef = get_embedding_function()

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef
        )
    except Exception:
        ingest_knowledge_base()
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef
        )

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunks.append({
                "content": doc,
                "source": meta.get("source", "unknown"),
                "topic": meta.get("topic", "unknown"),
                "relevance_score": round(1 - dist, 3)
            })
    return chunks


# ── Memory (past solved problems) ──────────────────────────────────────────────

def get_memory_collection():
    client = get_chroma_client()
    ef = get_embedding_function()
    return client.get_or_create_collection(
        name=MEMORY_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )


def store_solved_problem(problem_id: str, problem_text: str, solution: str,
                         topic: str, feedback: str = "correct"):
    """Persist a solved problem to memory."""
    collection = get_memory_collection()
    collection.upsert(
        documents=[problem_text],
        ids=[problem_id],
        metadatas=[{
            "solution": solution[:1000],   # truncate for meta limits
            "topic": topic,
            "feedback": feedback
        }]
    )


def retrieve_similar_problems(query: str, top_k: int = 2) -> list[dict]:
    """Find similar previously solved problems."""
    collection = get_memory_collection()
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    similar = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            similarity = round(1 - dist, 3)
            if similarity > 0.6:  # only return actually similar ones
                similar.append({
                    "problem": doc,
                    "solution_summary": meta.get("solution", ""),
                    "topic": meta.get("topic", ""),
                    "feedback": meta.get("feedback", ""),
                    "similarity": similarity
                })
    return similar