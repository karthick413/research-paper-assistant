
import ollama
import faiss
import numpy as np

# 1. Our sample document passages
documents = [
    "RAG retrieves relevant documents before generating an answer.",
    "A cat is a small domesticated animal that often lives with humans.",
]

# 2. Convert the passages into embeddings
response = ollama.embed(
    model="nomic-embed-text",
    input=documents,
)

# FAISS expects a NumPy array of float32 values
vectors = np.array(response["embeddings"], dtype=np.float32)

# 3. Normalize vectors so inner product acts like cosine similarity
faiss.normalize_L2(vectors)

# 4. Create a FAISS index for 768-dimensional vectors
dimension = vectors.shape[1]
index = faiss.IndexFlatIP(dimension)

# 5. Store our document vectors in the index
index.add(vectors)

print("Documents stored:", index.ntotal)
print("Vector dimensions:", index.d)

# 6. Search with a new question
query = "How does RAG find information before answering?"

query_response = ollama.embed(
    model="nomic-embed-text",
    input=query,
)

query_vector = np.array(
    query_response["embeddings"],
    dtype=np.float32,
)

faiss.normalize_L2(query_vector)

# Find the two most similar passages
scores, indices = index.search(query_vector, k=2)

print("\nSEARCH RESULTS")

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]), start=1
):
    print(f"\nRank: {rank}")
    print(f"Similarity score: {score:.4f}")
    print(f"Passage: {documents[idx]}")
