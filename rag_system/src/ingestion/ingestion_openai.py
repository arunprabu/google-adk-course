# This is naive RAG system (not multi modal RAG system)

# Step 0: Install the necessary pypi packages
# uv add chromadb pypdf python-dotenv openai
# Step 1: Load the PDF file and read the content
# Step 2: Break the file content into chunks (by following chunking strategy)
# Step 3: Setup embedding model (text-embedding-3-small is our pick - needs api key)
# Step 4: embedding model will return vector embeddings
# Step 5: setup chroma db, connect to it
# Step 6: create collection in chromadb and ingest the vector embeddings

import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]  # project root

# Chunking Strategy
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters


# Step 1: Load the PDF
def load_pdfs(folder: Path) -> list[dict]:
    pages = []
    for pdf in folder.glob("*.pdf"):
        for page_num, page in enumerate(PdfReader(pdf).pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "source": pdf.name, "page": page_num})
    return pages


# Step 2: Break the file content into chunks (by following chunking strategy)
def chunk_pages(pages: list[dict]) -> list[dict]:
    chunks = []
    for p in pages:
        chunk_num = 1
        for i in range(0, len(p["text"]), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = p["text"][i : i + CHUNK_SIZE].strip()
            if chunk:
                chunks.append(
                    {
                        "id": f"{p['source']}_page{p['page']}_chunk{chunk_num}",  # file1.pdf_page1_chunk_1
                        "text": chunk,
                        "metadata": {"source": p["source"], "page": p["page"]},
                    }
                )
                chunk_num += 1
    return chunks


# Step 3 & 4: Setup embedding model (text-embedding-3-small is our pick - needs api key)
embedding_fn = OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-3-small"
)

# Step 5: setup chroma db, connect to it (saved into my disk)
vector_db_client = chromadb.PersistentClient(path=str(ROOT / "chroma_db_openai"))

# Step 6: create collection in chromadb
collection = vector_db_client.get_or_create_collection(
    name="knowledge_base", embedding_function=embedding_fn
)

pages = load_pdfs(ROOT / "data" / "pdf")
chunks = chunk_pages(pages)

collection.upsert(
    ids=[c["id"] for c in chunks],
    documents=[c["text"] for c in chunks],  # original text
    metadatas=[c["metadata"] for c in chunks],
)

print(f"Loaded {len(pages)} Pages")
print(f"Total Chunks {len(chunks)} Chunks")
print(f"Total Chunks in DB: {collection.count()}")
