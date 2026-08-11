from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from collections import defaultdict

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
# Structured output
# --------------------------------------------------

class QueryVariations(BaseModel):
    queries: List[str]


# --------------------------------------------------
# Step 1: Generate query variations
# --------------------------------------------------

original_query = "How does Tesla make money?"

print(f"Original Query: {original_query}\n")

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
# Step 2: Retrieve documents for each query
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


# --------------------------------------------------
# Step 3: Reciprocal Rank Fusion
# --------------------------------------------------

def reciprocal_rank_fusion(
    chunk_lists,
    k=60,
    verbose=True
):
    """
    Combine multiple ranked retrieval lists using RRF.

    RRF score:
        score = 1 / (k + rank)
    """

    if verbose:
        print("\n" + "=" * 60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("=" * 60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")

    rrf_scores = defaultdict(float)

    all_unique_chunks = {}

    chunk_id_map = {}
    chunk_counter = 1

    # Process each query's ranked results
    for query_idx, chunks in enumerate(chunk_lists, 1):

        if verbose:
            print(f"Processing Query {query_idx} results:")

        for position, chunk in enumerate(chunks, 1):

            chunk_content = chunk.page_content

            # Assign ID to new chunks
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = (
                    f"Chunk_{chunk_counter}"
                )
                chunk_counter += 1

            chunk_id = chunk_id_map[chunk_content]

            # Store actual document
            all_unique_chunks[chunk_content] = chunk

            # RRF score
            position_score = 1 / (k + position)

            rrf_scores[chunk_content] += position_score

            if verbose:
                print(
                    f"  Position {position}: "
                    f"{chunk_id} "
                    f"+{position_score:.4f} "
                    f"(running total: "
                    f"{rrf_scores[chunk_content]:.4f})"
                )

                print(
                    f"    Preview: "
                    f"{chunk_content[:80]}..."
                )

        if verbose:
            print()

    # Sort by highest RRF score
    sorted_chunks = sorted(
        [
            (
                all_unique_chunks[chunk_content],
                score
            )
            for chunk_content, score
            in rrf_scores.items()
        ],
        key=lambda x: x[1],
        reverse=True
    )

    if verbose:
        print(
            f"✅ RRF Complete! "
            f"Processed {len(sorted_chunks)} "
            f"unique chunks from "
            f"{len(chunk_lists)} queries."
        )

    return sorted_chunks


# --------------------------------------------------
# Step 4: Apply RRF
# --------------------------------------------------

fused_results = reciprocal_rank_fusion(
    all_retrieval_results,
    k=60,
    verbose=True
)


# --------------------------------------------------
# Step 5: Display final ranking
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL RRF RANKING")
print("=" * 60)

print(
    f"\nTop {min(10, len(fused_results))} "
    f"documents after RRF fusion:\n"
)

for rank, (doc, rrf_score) in enumerate(
    fused_results[:10],
    1
):

    print(
        f"🏆 RANK {rank} "
        f"(RRF Score: {rrf_score:.4f})"
    )

    print(
        f"{doc.page_content[:200]}..."
    )

    print("-" * 50)


print(
    f"\n✅ RRF Complete! "
    f"Fused {len(fused_results)} unique "
    f"documents from "
    f"{len(query_variations)} query variations."
)

print("\n💡 Key benefits:")
print(
    "   • Documents appearing in multiple "
    "queries get boosted scores"
)
print(
    "   • Higher positions contribute more "
    "to the final score"
)
print(
    "   • Balanced fusion using k=60 "
    "for gentle position penalties"
)