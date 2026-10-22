
import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a helpful AI and machine learning tutor. "
                "Explain technical concepts clearly to beginners. "
                "If you know the answer, provide it directly."
            ),
        },
        {
            "role": "user",
            "content": (
                "What is Retrieval-Augmented Generation (RAG)? "
                "Explain how it works in three simple sentences."
            ),
        },
    ],
    options={"temperature": 0.2},
)

print("LLM RESPONSE:")
print(response["message"]["content"])


# Test the embedding model
embedding_response = ollama.embed(
    model="nomic-embed-text",
    input=[
        "RAG retrieves relevant documents before generating an answer.",
        "A cat is a small domesticated animal.",
    ],
)

vectors = embedding_response["embeddings"]

print("\nEMBEDDING TEST")
print("Number of vectors:", len(vectors))
print("Dimensions per vector:", len(vectors[0]))
print("First five values:", vectors[0][:5])
