"""
Simple RAG application using OpenAI and FAISS.
Expected project layout:
    project/
    ├── .env                 # OPENAI_API_KEY=...
    ├── data/                # Put .txt, .pdf, and .docx files here
    └── rag.py

Install dependencies in your existing environment:
    pip install openai python-dotenv numpy faiss-cpu pypdf python-docx

Run:
    python rag.py

The FAISS index and its chunk metadata are saved in ./vector_store.
The index is built only when either saved file is missing. To rebuild it
manually, stop the app and delete the vector_store folder.
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from docx import Document
from openai import OpenAI
from pypdf import PdfReader


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.json"

# Keep the embedding model inline, as requested.
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

CHUNK_SIZE = 1_000       # Approximate characters per chunk
CHUNK_OVERLAP = 150      # Characters repeated between neighbouring chunks
TOP_K = 4                # Number of chunks to retrieve for each question
EMBEDDING_BATCH_SIZE = 100

# Chat model parameters
TEMPERATURE = 0.2          # it control response creativity : lower values result in focused and predictable answers, while higher values lead to creative and random outputs
TOP_P = 1.0                # it sets a cumulative probability threshold ensuring the model only considers the most likely words up to that probablility
MAX_TOKENS = 700           
 

# ============================================================
# 2. CLIENT SETUP
# ============================================================

load_dotenv(BASE_DIR / ".env")
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY was not found. Add OPENAI_API_KEY=your_key "
        "to the .env file in the project root."
    )

client = OpenAI(api_key=api_key)


# ============================================================
# 3. DOCUMENT LOADING
# ============================================================

def load_documents():
    """Read supported text, PDF, and Word files from the data folder."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data folder not found: {DATA_DIR}\nCreate a 'data' folder "
            "beside rag.py and add .txt, .pdf, or .docx files."
        )

    documents = []

    for file_path in sorted(DATA_DIR.rglob("*")):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        text = ""

        try:
            if suffix == ".txt":
                text = file_path.read_text(encoding="utf-8", errors="ignore")

            elif suffix == ".pdf":
                reader = PdfReader(str(file_path))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)

            elif suffix == ".docx":
                doc = Document(str(file_path))
                text = "\n".join(
                    paragraph.text for paragraph in doc.paragraphs
                    if paragraph.text.strip()
                )

            else:
                # Ignore unsupported files, such as .doc, images, or spreadsheets.
                continue

        except Exception as exc:
            print(f"Warning: Could not read {file_path.name}: {exc}")
            continue

        text = text.strip()
        if text:
            documents.append({
                "source": str(file_path.relative_to(DATA_DIR)),
                "text": text,
            })
            print(f"Loaded: {file_path.relative_to(DATA_DIR)}")
        else:
            print(f"Skipped (no extractable text): {file_path.name}")

    if not documents:
        raise ValueError(
            "No readable text found in the data folder. Add a text-based "
            ".txt, .pdf, or .docx file. Scanned PDFs may need OCR first."
        )

    return documents


# ============================================================
# 4. SIMPLE CHUNKING
# ============================================================

def split_into_chunks(documents):
    """Split each document into overlapping character-based chunks."""
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for document in documents:
        text = document["text"]

        for start in range(0, len(text), step):
            chunk_text = text[start:start + CHUNK_SIZE].strip()
            if chunk_text:
                chunks.append({
                    "source": document["source"],
                    "text": chunk_text,
                })

    if not chunks:
        raise ValueError("Chunking produced no text chunks.")

    print(f"Created {len(chunks)} chunks.")
    return chunks


# ============================================================
# 5. OPENAI EMBEDDINGS
# ============================================================

def create_embeddings(texts):
    """Generate embedding vectors in batches using OpenAI."""
    all_embeddings = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        # Keep the returned vectors in the same order as the input texts.
        batch_embeddings = [
            item.embedding for item in sorted(response.data, key=lambda item: item.index)
        ]
        all_embeddings.extend(batch_embeddings)

    vectors = np.asarray(all_embeddings, dtype=np.float32)

    if vectors.ndim != 2 or len(vectors) != len(texts):
        raise RuntimeError("Embedding response did not match the input text count.")

    # Normalize vectors so inner-product search behaves like cosine similarity.
    faiss.normalize_L2(vectors)
    return vectors


# ============================================================
# 6. BUILD AND SAVE THE FAISS VECTOR DATABASE
# ============================================================

def build_vector_store():
    """Load documents, chunk them, embed them, and save the FAISS index."""
    documents = load_documents()
    chunks = split_into_chunks(documents)

    print("Creating embeddings. This may take a little while...")
    vectors = create_embeddings([chunk["text"] for chunk in chunks])

    # IndexFlatIP uses inner product; normalized vectors make this cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    with CHUNKS_PATH.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print(f"Saved FAISS index to: {INDEX_PATH}")
    print(f"Saved chunk metadata to: {CHUNKS_PATH}")
    return index, chunks


def load_or_build_vector_store():
    """Reuse the saved vector store when both required files exist."""
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    if INDEX_PATH.exists() and CHUNKS_PATH.exists():
        print("Found saved vector store. Loading it without rebuilding...")
        index = faiss.read_index(str(INDEX_PATH))

        with CHUNKS_PATH.open("r", encoding="utf-8") as file:
            chunks = json.load(file)

        if index.ntotal != len(chunks):
            print("Saved index and chunk metadata do not match. Rebuilding...")
            return build_vector_store()

        if index.ntotal == 0:
            print("Saved vector store is empty. Rebuilding...")
            return build_vector_store()

        print(f"Loaded {len(chunks)} chunks from the saved vector store.")
        return index, chunks

    print("No complete saved vector store found. Building it now...")
    return build_vector_store()


# ============================================================
# 7. RETRIEVAL
# ============================================================

def retrieve_relevant_chunks(question, index, chunks):
    """Find the most relevant saved chunks for the user's question."""
    query_vector = create_embeddings([question])
    number_to_fetch = min(TOP_K, index.ntotal)

    scores, positions = index.search(query_vector, number_to_fetch)
    results = []

    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue

        chunk = chunks[int(position)]
        results.append({
            "source": chunk["source"],
            "text": chunk["text"],
            "score": float(score),
        })

    return results


# ============================================================
# 8. ANSWER GENERATION
# ============================================================

def answer_question(question, index, chunks):
    """Answer using retrieved document text and show the source filenames."""
    retrieved_chunks = retrieve_relevant_chunks(question, index, chunks)

    if not retrieved_chunks:
        return "I couldn't find relevant information in the indexed documents."

    context_parts = []
    for number, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {number}: {chunk['source']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant answering questions about the "
                    "provided documents. Use the supplied context as your primary "
                    "source. If the context does not contain the answer, say you "
                    "could not find the answer in the provided documents. Do not "
                    "invent facts. Mention source filenames when useful."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Retrieved document context:\n{context}"
                ),
            },
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )

    answer = response.choices[0].message.content
    answer = answer.strip() if answer else "The model returned an empty answer."

    sources = []
    for chunk in retrieved_chunks:
        if chunk["source"] not in sources:
            sources.append(chunk["source"])

    return answer + "\n\nSources: " + ", ".join(sources)


# ============================================================
# 9. COMMAND-LINE CHAT LOOP
# ============================================================

def main():
    """Load the vector store once and answer questions until the user exits."""
    try:
        index, chunks = load_or_build_vector_store()
    except Exception as exc:
        print(f"\nCould not prepare the vector store: {exc}")
        raise

    print("\nSimple RAG chatbot is ready.")
    print("Ask a question about your documents, or type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            print("\nAssistant:")
            print(answer_question(question, index, chunks))
            print()
        except Exception as exc:
            print(f"\nSorry, the question could not be answered: {exc}\n")


if __name__ == "__main__":
    main()
