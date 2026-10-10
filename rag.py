"""Persistent similarity search over the supplied policy documents."""

import argparse
import hashlib
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from policy_loader import POLICY_FILE, load_policy_documents

ROOT = Path(__file__).resolve().parent
EMBEDDING_MODEL = "nomic-embed-text:latest"
documents = load_policy_documents()


def base_url():
    return os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


def open_store(directory=None):
    fingerprint = hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest()
    embedding = OllamaEmbeddings(
        model=EMBEDDING_MODEL, base_url=base_url(), num_gpu=99,
        num_thread=2, num_ctx=1024, keep_alive=120,
        client_kwargs={"timeout": 120, "trust_env": False},
    )
    return Chroma(
        collection_name="university_policies_" + fingerprint[:16],
        persist_directory=str(directory or ROOT / ".index"),
        embedding_function=embedding,
        collection_metadata={"policy_sha256": fingerprint, "embedding_model": EMBEDDING_MODEL},
    )


def missing_documents(store):
    stored = store.get(include=["documents", "metadatas"])
    expected = {d.metadata["policy_id"]: d for d in documents}
    for doc_id, body, metadata in zip(stored["ids"], stored["documents"], stored["metadatas"]):
        if doc_id not in expected or body != expected[doc_id].page_content or metadata != expected[doc_id].metadata:
            raise ValueError("Index contents do not match the supplied policies. Use a new index directory.")
    present = set(stored["ids"])
    return [doc for doc in documents if doc.metadata["policy_id"] not in present]


def build_index(store):
    missing = missing_documents(store)
    for start in range(0, len(missing), 16):
        batch = missing[start:start + 16]
        store.add_documents(batch, ids=[doc.metadata["policy_id"] for doc in batch])
        print(f"Indexed {len(documents) - len(missing) + start + len(batch)}/{len(documents)} policies", flush=True)
    if missing_documents(store):
        raise RuntimeError("The index is incomplete")
    return store


def retrieve(question, store):
    if not question.strip():
        raise ValueError("Enter a question")
    if missing_documents(store):
        raise RuntimeError("Build the policy index before asking questions")
    return store.similarity_search(question, k=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path)
    args = parser.parse_args()
    build_index(open_store(args.index_dir))


if __name__ == "__main__":
    main()
