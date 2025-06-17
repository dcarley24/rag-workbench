import json
import os
import time
import faiss
import numpy as np
from openai import OpenAI

client = OpenAI()  # Uses OPENAI_API_KEY from env
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100 # Process 100 chunks per API call

def load_chunks(filename="tagged_chunks.json"):
    with open(filename, "r") as f:
        return json.load(f)

def main():
    print("Loading chunks...")
    chunks = load_chunks()
    # Ensure embedding dimension is correct, default to 1536 for text-embedding-3-small
    dim = 1536
    index = faiss.IndexFlatL2(dim)
    chunks_with_vectors = []

    print(f"Embedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        batch_texts = [chunk["content"] for chunk in batch_chunks]

        try:
            # Make a single API call for the entire batch
            response = client.embeddings.create(
                input=batch_texts,
                model=EMBED_MODEL
            )

            # Get the embeddings for the whole batch
            batch_embeddings = [item.embedding for item in response.data]

            # Add the batch to the FAISS index
            index.add(np.array(batch_embeddings, dtype='float32'))

            # Process each chunk from the batch with its new embedding
            for j, chunk in enumerate(batch_chunks):
                chunk_record = {
                    "id": len(chunks_with_vectors), # Assign a new sequential ID
                    "title": chunk.get("title"),
                    "timestamp": chunk.get("timestamp"),
                    "role": chunk.get("role"),
                    "tags": chunk.get("tags"),
                    "content": chunk.get("content"),
                    "embedding": batch_embeddings[j]
                }
                chunks_with_vectors.append(chunk_record)

            print(f"Embedded batch {i // BATCH_SIZE + 1} of {len(chunks) // BATCH_SIZE + 1}...")

        except Exception as e:
            print(f"Error processing batch starting at chunk {i}: {e}")

        # Be kind to the API and avoid hitting rate limits
        time.sleep(1)

    print("Saving FAISS index and JSON metadata...")
    faiss.write_index(index, "faiss_index.faiss")
    with open("chunks_with_vectors.json", "w") as f:
        json.dump(chunks_with_vectors, f, indent=2)

    print("Embedding complete.")

if __name__ == "__main__":
    main()
