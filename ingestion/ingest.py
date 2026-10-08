import os
import sqlite3

import fitz
import faiss
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------

PDF_PATH = "data/document.pdf"
DB_PATH = "data/rag.db"
INDEX_PATH = "data/faiss.index"

MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# -----------------------------
# 1. Extract text from PDF
# -----------------------------

def extract_text_from_pdf(pdf_path):

    document = fitz.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text() + "\n"

    document.close()

    return text


# -----------------------------
# 2. Split text into chunks
# -----------------------------

def create_chunks(text):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + CHUNK_SIZE

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# -----------------------------
# 3. Store chunks in SQLite
# -----------------------------

def store_chunks(chunks):

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL
        )
    """)

    cursor.execute("DELETE FROM chunks")

    for chunk in chunks:

        cursor.execute(
            "INSERT INTO chunks (text) VALUES (?)",
            (chunk,)
        )

    connection.commit()

    connection.close()


# -----------------------------
# 4. Create embeddings
# -----------------------------

def create_embeddings(chunks):

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    return embeddings


# -----------------------------
# 5. Create FAISS index
# -----------------------------

def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    faiss.write_index(
        index,
        INDEX_PATH
    )


# -----------------------------
# Main ingestion pipeline
# -----------------------------

def main():

    os.makedirs("data", exist_ok=True)

    if not os.path.exists(PDF_PATH):

        print(f"PDF not found: {PDF_PATH}")
        return

    print("Extracting text from PDF...")

    text = extract_text_from_pdf(PDF_PATH)

    print("Creating chunks...")

    chunks = create_chunks(text)

    print(f"Created {len(chunks)} chunks.")

    print("Storing chunks in SQLite...")

    store_chunks(chunks)

    print("Creating embeddings...")

    embeddings = create_embeddings(chunks)

    print("Creating FAISS index...")

    create_faiss_index(embeddings)

    print("Ingestion completed successfully.")


if __name__ == "__main__":
    main()
