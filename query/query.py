import os
import sqlite3

import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# -----------------------------
# Configuration
# -----------------------------

DB_PATH = "data/rag.db"
INDEX_PATH = "data/faiss.index"

MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"

TOP_K = 5


# -----------------------------
# Load environment variables
# -----------------------------

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# -----------------------------
# Load embedding model
# -----------------------------

embedding_model = SentenceTransformer(MODEL_NAME)


# -----------------------------
# Load FAISS index
# -----------------------------

index = faiss.read_index(INDEX_PATH)


# -----------------------------
# 1. Retrieve relevant chunks
# -----------------------------

def retrieve_chunks(query):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_embedding,
        TOP_K
    )

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    chunks = []

    for index_id in indices[0]:

        cursor.execute(
            "SELECT text FROM chunks WHERE id = ?",
            (int(index_id) + 1,)
        )

        result = cursor.fetchone()

        if result:
            chunks.append(result[0])

    connection.close()

    return chunks


# -----------------------------
# 2. Build prompt
# -----------------------------

def build_prompt(query, chunks):

    context = "\n\n".join(chunks)

    prompt = f"""
You are a helpful assistant answering questions
using the provided document context.

Answer the question using only the information
provided in the context.

If the answer cannot be found in the context,
say that you could not find the answer in the
provided document.

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt


# -----------------------------
# 3. Generate answer using Gemini
# -----------------------------

def generate_answer(prompt):

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text


# -----------------------------
# Main query pipeline
# -----------------------------

def main():

    print("Enterprise Document RAG")
    print("-----------------------")

    while True:

        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        print("\nRetrieving relevant information...")

        chunks = retrieve_chunks(query)

        print(f"Retrieved {len(chunks)} relevant chunks.")

        prompt = build_prompt(
            query,
            chunks
        )

        print("\nGenerating answer...")

        answer = generate_answer(prompt)

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()
