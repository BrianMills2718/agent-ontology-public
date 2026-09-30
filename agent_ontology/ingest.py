#!/usr/bin/env python3
"""
ao-ingest: Ingest documents into a ChromaDB vector store for RAG agents.

Reads text files, splits into chunks, generates embeddings, and stores
in a ChromaDB persistent collection.

Usage:
    ao-ingest docs/ --store my_kb
    ao-ingest paper.txt --store research --chunk-size 512
    ao-ingest docs/*.md --store docs --overlap 64
    ao-ingest --list                          # List existing stores
    ao-ingest --stats my_kb                   # Show store statistics
"""

import argparse
import glob
import hashlib
import os
import sys
import time


DEFAULT_STORE_DIR = os.path.expanduser("~/.agent_ontology/stores")
DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 64


def chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def read_file(path):
    """Read a text file, returning its content. Supports .txt, .md, .py, etc."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        print(f"  Warning: could not read {path}: {e}")
        return None


def compute_id(text, source, index):
    """Generate a deterministic ID for a chunk."""
    h = hashlib.md5(f"{source}:{index}:{text[:100]}".encode()).hexdigest()[:12]
    return f"{os.path.basename(source)}_{index}_{h}"


def ingest(paths, store_name, store_dir=DEFAULT_STORE_DIR,
           chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP,
           embedding_model="text-embedding-3-small", embedding_provider="openai",
           verbose=True):
    """Ingest files into a ChromaDB collection.

    Args:
        paths: List of file paths or glob patterns
        store_name: Name of the ChromaDB collection
        store_dir: Directory for ChromaDB persistent storage
        chunk_size: Characters per chunk
        overlap: Overlap between chunks
        embedding_model: Embedding model name
        embedding_provider: Provider (openai, google, huggingface)
        verbose: Print progress

    Returns:
        dict with ingestion statistics
    """
    try:
        import chromadb
    except ImportError:
        print("Error: chromadb not installed. Run: pip install chromadb")
        sys.exit(1)

    os.makedirs(store_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=store_dir)

    # Get or create collection
    collection = client.get_or_create_collection(
        name=store_name,
        metadata={"embedding_model": embedding_model,
                  "embedding_provider": embedding_provider}
    )

    # Resolve file paths
    all_files = []
    for p in paths:
        if '*' in p or '?' in p:
            all_files.extend(glob.glob(p, recursive=True))
        elif os.path.isdir(p):
            for ext in ('*.txt', '*.md', '*.py', '*.json', '*.yaml', '*.yml',
                        '*.rst', '*.html', '*.csv'):
                all_files.extend(glob.glob(os.path.join(p, '**', ext), recursive=True))
        elif os.path.isfile(p):
            all_files.append(p)

    all_files = sorted(set(all_files))
    if not all_files:
        print("No files found to ingest.")
        return {"files": 0, "chunks": 0}

    if verbose:
        print(f"\nIngesting {len(all_files)} file(s) into '{store_name}'")
        print(f"  Store dir: {store_dir}")
        print(f"  Chunk size: {chunk_size}, overlap: {overlap}")
        print(f"  Embedding: {embedding_model} ({embedding_provider})")

    # Generate embeddings
    embed_fn = _get_embed_fn(embedding_model, embedding_provider)

    total_chunks = 0
    t0 = time.time()

    for fpath in all_files:
        text = read_file(fpath)
        if not text:
            continue

        chunks = chunk_text(text, chunk_size, overlap)
        if not chunks:
            continue

        if verbose:
            print(f"  {os.path.basename(fpath)}: {len(chunks)} chunks")

        # Batch add to ChromaDB
        ids = [compute_id(c, fpath, i) for i, c in enumerate(chunks)]
        metadatas = [{"source": fpath, "chunk_index": i} for i in range(len(chunks))]

        if embed_fn:
            embeddings = embed_fn(chunks)
            collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        else:
            collection.upsert(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
            )

        total_chunks += len(chunks)

    elapsed = time.time() - t0

    stats = {
        "files": len(all_files),
        "chunks": total_chunks,
        "collection": store_name,
        "total_in_store": collection.count(),
        "elapsed_sec": round(elapsed, 1),
    }

    if verbose:
        print(f"\n  Done: {stats['chunks']} chunks from {stats['files']} files "
              f"({stats['elapsed_sec']}s)")
        print(f"  Total in '{store_name}': {stats['total_in_store']} chunks")

    return stats


def _get_embed_fn(model, provider):
    """Return an embedding function or None (let ChromaDB use its default)."""
    if provider == "openai":
        try:
            import openai
            client = openai.OpenAI()

            def embed(texts):
                resp = client.embeddings.create(input=texts, model=model)
                return [d.embedding for d in resp.data]
            return embed
        except Exception:
            return None
    elif provider == "google":
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

            def embed(texts):
                result = genai.embed_content(model=model, content=texts)
                return result["embedding"]
            return embed
        except Exception:
            return None
    elif provider == "huggingface":
        try:
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer(model)

            def embed(texts):
                return st.encode(texts).tolist()
            return embed
        except Exception:
            return None

    return None


def list_stores(store_dir=DEFAULT_STORE_DIR):
    """List all ChromaDB collections in the store directory."""
    try:
        import chromadb
    except ImportError:
        print("Error: chromadb not installed.")
        return

    if not os.path.exists(store_dir):
        print(f"No stores found at {store_dir}")
        return

    client = chromadb.PersistentClient(path=store_dir)
    collections = client.list_collections()

    if not collections:
        print("No collections found.")
        return

    print(f"\nStores in {store_dir}:")
    for c in collections:
        count = c.count()
        meta = c.metadata or {}
        print(f"  {c.name}: {count} chunks "
              f"(model: {meta.get('embedding_model', 'default')})")


def show_stats(store_name, store_dir=DEFAULT_STORE_DIR):
    """Show statistics for a specific collection."""
    try:
        import chromadb
    except ImportError:
        print("Error: chromadb not installed.")
        return

    client = chromadb.PersistentClient(path=store_dir)
    try:
        collection = client.get_collection(store_name)
    except Exception:
        print(f"Collection '{store_name}' not found.")
        return

    count = collection.count()
    meta = collection.metadata or {}

    print(f"\nCollection: {store_name}")
    print(f"  Chunks: {count}")
    print(f"  Embedding model: {meta.get('embedding_model', 'default')}")
    print(f"  Provider: {meta.get('embedding_provider', 'default')}")

    if count > 0:
        # Sample a few items to show sources
        sample = collection.peek(limit=min(5, count))
        sources = set()
        for m in (sample.get("metadatas") or []):
            if m and m.get("source"):
                sources.add(m["source"])
        if sources:
            print(f"  Sources (sample): {len(sources)} files")
            for s in sorted(sources)[:5]:
                print(f"    {s}")


def main():
    parser = argparse.ArgumentParser(
        description="ao-ingest: Ingest documents into ChromaDB for RAG agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="*", help="Files or directories to ingest")
    parser.add_argument("--store", "-s", help="Collection name")
    parser.add_argument("--store-dir", default=DEFAULT_STORE_DIR,
                        help=f"ChromaDB storage directory (default: {DEFAULT_STORE_DIR})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Characters per chunk (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                        help=f"Overlap between chunks (default: {DEFAULT_OVERLAP})")
    parser.add_argument("--embedding-model", default="text-embedding-3-small",
                        help="Embedding model (default: text-embedding-3-small)")
    parser.add_argument("--embedding-provider", default="openai",
                        choices=["openai", "google", "huggingface"],
                        help="Embedding provider (default: openai)")
    parser.add_argument("--list", action="store_true",
                        help="List existing stores")
    parser.add_argument("--stats", metavar="STORE",
                        help="Show statistics for a store")
    parser.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args()

    if args.list:
        list_stores(args.store_dir)
        return

    if args.stats:
        show_stats(args.stats, args.store_dir)
        return

    if not args.paths:
        parser.error("No files specified. Use --list to see existing stores.")

    if not args.store:
        parser.error("--store NAME is required when ingesting files.")

    ingest(
        paths=args.paths,
        store_name=args.store,
        store_dir=args.store_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_model=args.embedding_model,
        embedding_provider=args.embedding_provider,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
