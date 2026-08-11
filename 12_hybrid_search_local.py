from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv

import os

load_dotenv()

# --------------------------------------------------
# Setup
# --------------------------------------------------

PERSISTENT_DIRECTORY = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=PERSISTENT_DIRECTORY,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# --------------------------------------------------
# Load documents from Chroma
# --------------------------------------------------

print("Loading documents from ChromaDB...")

stored_data = vectorstore.get(
    include=["documents", "metadatas"]
)

documents = []

for content, metadata in zip(
    stored_data["documents"],
    stored_data["metadatas"]
):
    documents.append(
        Document(
            page_content=content,
            metadata=metadata or {}
        )
    )

print(f"Loaded {len(documents)} chunks from ChromaDB.\n")


# --------------------------------------------------
# Vector Retriever
# --------------------------------------------------

print("Setting up Vector Retriever...")

vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)


# --------------------------------------------------
# BM25 Retriever
# --------------------------------------------------

print("Setting up BM25 Retriever...")

bm25_retriever = BM25Retriever.from_documents(
    documents
)

bm25_retriever.k = 5


# --------------------------------------------------
# Hybrid Search
# --------------------------------------------------

def hybrid_search(query, vector_weight=0.7, bm25_weight=0.3):

    print("\n" + "=" * 60)
    print(f"HYBRID SEARCH")
    print("=" * 60)

    print(f"\nQuery: {query}")
    print(f"Vector weight: {vector_weight}")
    print(f"BM25 weight: {bm25_weight}")

    # Vector results
    vector_docs = vector_retriever.invoke(query)

    # BM25 results
    bm25_docs = bm25_retriever.invoke(query)

    print("\n--- Vector Search Results ---")

    for i, doc in enumerate(vector_docs, 1):
        print(f"\n{i}. {doc.page_content[:200]}...")

    print("\n--- BM25 Keyword Results ---")

    for i, doc in enumerate(bm25_docs, 1):
        print(f"\n{i}. {doc.page_content[:200]}...")

    # --------------------------------------------------
    # Weighted rank fusion
    # --------------------------------------------------

    scores = {}
    unique_docs = {}

    # Vector ranking
    for rank, doc in enumerate(vector_docs, 1):

        doc_id = doc.page_content

        unique_docs[doc_id] = doc

        scores[doc_id] = scores.get(doc_id, 0) + (
            vector_weight / rank
        )

    # BM25 ranking
    for rank, doc in enumerate(bm25_docs, 1):

        doc_id = doc.page_content

        unique_docs[doc_id] = doc

        scores[doc_id] = scores.get(doc_id, 0) + (
            bm25_weight / rank
        )

    # Sort highest score first
    ranked_results = sorted(
        unique_docs.items(),
        key=lambda item: scores[item[0]],
        reverse=True
    )

    return [
        (doc, scores[doc_id])
        for doc_id, doc in ranked_results
    ]


# --------------------------------------------------
# Test Query
# --------------------------------------------------

query = "How does Tesla make money?"

hybrid_results = hybrid_search(query)


# --------------------------------------------------
# Display final hybrid ranking
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL HYBRID SEARCH RANKING")
print("=" * 60)

for rank, (doc, score) in enumerate(
    hybrid_results[:10],
    1
):

    print(
        f"\n🏆 Rank {rank} "
        f"(Hybrid Score: {score:.4f})"
    )

    print(doc.page_content[:300])

    print("-" * 60)


print("\n✅ Hybrid Search Complete!")