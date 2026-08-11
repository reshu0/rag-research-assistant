from collections import defaultdict
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder


load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

PERSISTENT_DIRECTORY = "db/chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama3.2:3b"

VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3

RETRIEVAL_K = 5
RRF_K = 60
RERANK_CANDIDATES = 15
FINAL_DOCUMENTS = 5


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("Loading ChromaDB...")

vectorstore = Chroma(
    persist_directory=PERSISTENT_DIRECTORY,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

print("Loading Ollama...")

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
)

print("Loading cross-encoder reranker...")

reranker = CrossEncoder(RERANKER_MODEL)


# ============================================================
# LOAD DOCUMENTS FROM CHROMADB
# ============================================================

print("Loading documents from ChromaDB...")

stored_data = vectorstore.get(
    include=["documents", "metadatas"]
)

documents = []

for content, metadata in zip(
    stored_data["documents"],
    stored_data["metadatas"],
):
    documents.append(
        Document(
            page_content=content,
            metadata=metadata or {},
        )
    )

print(f"Loaded {len(documents)} chunks.")


# ============================================================
# RETRIEVERS
# ============================================================

vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": RETRIEVAL_K}
)

bm25_retriever = BM25Retriever.from_documents(
    documents
)

bm25_retriever.k = RETRIEVAL_K


# ============================================================
# QUERY VARIATIONS
# ============================================================

class QueryVariations(BaseModel):
    queries: List[str]


def generate_query_variations(query: str) -> List[str]:

    print("\n" + "=" * 70)
    print("STEP 1: MULTI-QUERY GENERATION")
    print("=" * 70)

    structured_llm = llm.with_structured_output(
        QueryVariations
    )

    prompt = f"""
Generate exactly 3 different search queries for the
following user question.

Original question:
{query}

Each query should approach the question from a different
angle while preserving the original intent.

Return only the 3 search queries.
"""

    response = structured_llm.invoke(prompt)

    variations = response.queries[:3]

    print("\nGenerated queries:")

    for i, variation in enumerate(variations, 1):
        print(f"{i}. {variation}")

    return variations


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(query: str) -> List[Document]:

    vector_docs = vector_retriever.invoke(query)

    bm25_docs = bm25_retriever.invoke(query)

    scores = defaultdict(float)
    unique_docs = {}

    # --------------------------------------------------------
    # Vector ranking
    # --------------------------------------------------------

    for rank, doc in enumerate(vector_docs, 1):

        doc_id = doc.page_content

        unique_docs[doc_id] = doc

        scores[doc_id] += (
            VECTOR_WEIGHT / rank
        )

    # --------------------------------------------------------
    # BM25 ranking
    # --------------------------------------------------------

    for rank, doc in enumerate(bm25_docs, 1):

        doc_id = doc.page_content

        unique_docs[doc_id] = doc

        scores[doc_id] += (
            BM25_WEIGHT / rank
        )

    # --------------------------------------------------------
    # Sort by hybrid score
    # --------------------------------------------------------

    ranked = sorted(
        unique_docs.items(),
        key=lambda item: scores[item[0]],
        reverse=True,
    )

    return [
        doc
        for doc_id, doc in ranked
    ]


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    result_lists: List[List[Document]],
    k: int = RRF_K,
) -> List[Document]:

    scores = defaultdict(float)
    unique_docs = {}

    for results in result_lists:

        for rank, doc in enumerate(
            results,
            start=1,
        ):

            doc_id = doc.page_content

            unique_docs[doc_id] = doc

            scores[doc_id] += (
                1 / (k + rank)
            )

    ranked = sorted(
        unique_docs.items(),
        key=lambda item: scores[item[0]],
        reverse=True,
    )

    return [
        doc
        for doc_id, doc in ranked
    ]


# ============================================================
# CROSS-ENCODER RERANKING
# ============================================================

def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = FINAL_DOCUMENTS,
):

    if not documents:
        return []

    pairs = [
        [query, doc.page_content]
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked[:top_n]


# ============================================================
# MAIN RAG PIPELINE
# ============================================================

def run_rag(query: str) -> str:

    print("\n" + "=" * 70)
    print("USER QUERY")
    print("=" * 70)

    print(query)

    # --------------------------------------------------------
    # STEP 1: Multi-query generation
    # --------------------------------------------------------

    query_variations = generate_query_variations(
        query
    )

    # Include original query
    all_queries = [
        query
    ] + query_variations

    # --------------------------------------------------------
    # STEP 2: Hybrid retrieval
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 2: HYBRID RETRIEVAL")
    print("=" * 70)

    retrieval_results = []

    for i, search_query in enumerate(
        all_queries,
        1,
    ):

        results = hybrid_search(
            search_query
        )

        retrieval_results.append(
            results
        )

        print(
            f"Query {i}: "
            f"retrieved {len(results)} candidates"
        )

    # --------------------------------------------------------
    # STEP 3: Reciprocal Rank Fusion
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 3: RECIPROCAL RANK FUSION")
    print("=" * 70)

    fused_documents = reciprocal_rank_fusion(
        retrieval_results
    )

    candidates = fused_documents[
        :RERANK_CANDIDATES
    ]

    print(
        f"RRF produced "
        f"{len(fused_documents)} unique documents."
    )

    print(
        f"Sending top {len(candidates)} candidates "
        f"to reranker."
    )

    # --------------------------------------------------------
    # STEP 4: Cross-encoder reranking
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 4: CROSS-ENCODER RERANKING")
    print("=" * 70)

    reranked = rerank_documents(
        query,
        candidates,
        top_n=FINAL_DOCUMENTS,
    )

    for rank, (doc, score) in enumerate(
        reranked,
        1,
    ):

        print(
            f"\nRank {rank} "
            f"| Score: {score:.4f}"
        )

        print(
            doc.page_content[:300]
        )

    # --------------------------------------------------------
    # STEP 5: Build final context
    # --------------------------------------------------------

    top_documents = [
        doc
        for doc, score in reranked
    ]

    context = "\n\n".join(
        [
            f"[Source {i}]\n{doc.page_content}"
            for i, doc in enumerate(
                top_documents,
                1,
            )
        ]
    )

    # --------------------------------------------------------
    # STEP 6: LLM generation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 5: LLM GENERATION")
    print("=" * 70)

    prompt = f"""
You are a precise document question-answering assistant.

Answer the user's question using ONLY the provided documents.

Rules:
1. If the documents explicitly contain the answer,
   state it directly and confidently.
2. Do not say that information is missing when the answer
   is clearly present in the documents.
3. Do not invent facts or use outside knowledge.
4. If multiple documents support the answer, combine them
   when useful.
5. If the documents genuinely do not contain enough
   information, say:
   "I don't have enough information to answer that based
   on the provided documents."
6. Keep the answer concise and factual.

User question:
{query}

Retrieved documents:
{context}

Answer:
"""

    response = llm.invoke(prompt)

    answer = response.content

    # --------------------------------------------------------
    # STEP 7: Display answer
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(answer)

    # --------------------------------------------------------
    # STEP 8: Display sources
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for i, doc in enumerate(
        top_documents,
        1,
    ):

        source = doc.metadata.get(
            "source",
            "Unknown",
        )

        print(
            f"{i}. {source}"
        )

    return answer


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_query = "How much did Microsoft pay to acquire GitHub?"

    run_rag(test_query)