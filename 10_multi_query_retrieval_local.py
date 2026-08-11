from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

load_dotenv()

# --------------------------------------------------
# Setup
# --------------------------------------------------

persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)


# --------------------------------------------------
# Structured output for query variations
# --------------------------------------------------

class QueryVariations(BaseModel):
    queries: List[str]


# --------------------------------------------------
# Original query
# --------------------------------------------------

original_query = "How does Tesla make money?"

print(f"Original Query: {original_query}\n")


# --------------------------------------------------
# Step 1: Generate multiple query variations
# --------------------------------------------------

llm_with_structured_output = llm.with_structured_output(
    QueryVariations
)

prompt = f"""
Generate 3 different variations of this query that would
help retrieve relevant documents.

Original query: {original_query}

Return 3 alternative queries that rephrase or approach
the same question from different angles.
"""

response = llm_with_structured_output.invoke(prompt)

query_variations = response.queries

print("Generated Query Variations:")

for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

print("\n" + "=" * 60)


# --------------------------------------------------
# Step 2: Retrieve documents for every query
# --------------------------------------------------

retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

all_retrieval_results = []

for i, query in enumerate(query_variations, 1):

    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")

    docs = retriever.invoke(query)

    all_retrieval_results.append(docs)

    print(f"Retrieved {len(docs)} documents:\n")

    for j, doc in enumerate(docs, 1):

        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")

    print("-" * 50)


print("\n" + "=" * 60)
print("Multi-Query Retrieval Complete!")