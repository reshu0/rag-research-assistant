from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

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
# Load chunks from ChromaDB
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

print(f"Loaded {len(documents)} chunks.\n")


# --------------------------------------------------
# Vector Retriever
# --------------------------------------------------

vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10}
)


# --------------------------------------------------
# BM25 Retriever
# --------------------------------------------------

bm25_retriever = BM25Retriever.from_documents(
    documents
)

bm25_retriever.k = 10


# --------------------------------------------------
# Hybrid candidate retrieval
# --------------------------------------------------

def hybrid_candidates(query):
    """
    Retrieve candidate documents using both
    semantic vector search and BM25 keyword search.
    """

    vector_docs = vector_retriever.invoke(query)
    bm25_docs = bm25_retriever.invoke(query)

    # Combine while removing duplicates
    unique_docs = {}
    
    for doc in vector_docs + bm25_docs:
        unique_docs[doc.page_content] = doc

    return list(unique_docs.values())


# --------------------------------------------------
# Cross-Encoder Reranker
# --------------------------------------------------

print("Loading local cross-encoder reranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker loaded.\n")


# --------------------------------------------------
# Reranking
# --------------------------------------------------

def rerank_documents(query, documents, top_n=5):

    if not documents:
        return []

    pairs = [
        [query, doc.page_content]
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    scored_documents = list(
        zip(documents, scores)
    )

    scored_documents.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return scored_documents[:top_n]


# --------------------------------------------------
# Test query
# --------------------------------------------------

query = "How does Tesla make money?"

print("=" * 60)
print("QUERY")
print("=" * 60)

print(query)


# --------------------------------------------------
# Step 1: Hybrid retrieval
# --------------------------------------------------

print("\n" + "=" * 60)
print("STEP 1: HYBRID RETRIEVAL")
print("=" * 60)

candidate_docs = hybrid_candidates(query)

print(
    f"\nRetrieved {len(candidate_docs)} "
    "unique candidate chunks.\n"
)

for i, doc in enumerate(candidate_docs, 1):

    print(
        f"{i}. "
        f"{doc.page_content[:200]}..."
    )


# --------------------------------------------------
# Step 2: Cross-encoder reranking
# --------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2: CROSS-ENCODER RERANKING")
print("=" * 60)

reranked_docs = rerank_documents(
    query,
    candidate_docs,
    top_n=5
)

for rank, (doc, score) in enumerate(
    reranked_docs,
    1
):

    print(
        f"\n🏆 Rank {rank}"
        f" | Reranker Score: {score:.4f}"
    )

    print(doc.page_content[:400])

    print("-" * 60)


# --------------------------------------------------
# Final context
# --------------------------------------------------

top_documents = [
    doc
    for doc, score in reranked_docs
]

print("\n" + "=" * 60)
print("FINAL CONTEXT FOR LLM")
print("=" * 60)

for i, doc in enumerate(top_documents, 1):

    print(
        f"\nDocument {i}:"
    )

    print(
        doc.page_content[:500]
    )


print("\n" + "=" * 60)
print("✅ RERANKING COMPLETE")
print("=" * 60)