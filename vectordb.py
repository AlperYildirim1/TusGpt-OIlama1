import requests
import json
import chromadb
from chromadb.config import Settings
import os
import asyncio
import fitz  # PyMuPDF for PDF processing
import aiohttp

# ChromaDB client initialization
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db",
))

#EMBEDDING_API_URL = "http://host.docker.internal:8081/embed"
EMBEDDING_API_URL = "http://127.0.0.1:8081/embed"
HEADERS = {"Content-Type": "application/json"}

async def get_embeddings(text_data):
    """
    Sends text data to the embedding API and returns the generated embeddings asynchronously.
    """
    data = {"inputs": text_data}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                EMBEDDING_API_URL,
                json=data,
                headers=HEADERS
            ) as response:
                response.raise_for_status()  # Raise exception for 4xx/5xx status codes
                return await response.json()
                
    except aiohttp.ClientError as e:
        print(f"⚠️ Error fetching embeddings: {str(e)}")
        return None

async def rerank(query, texts):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                #"http://host.docker.internal:8082/rerank",
                "http://127.0.0.1:8082/rerank",  
                json={"query": query, "texts": texts}
            ) as response:
                response.raise_for_status()
                return await response.json()
    except aiohttp.ClientError as e:
        print(f"Request failed: {e}")
        return None

async def split_text_into_chunks(text, chunk_size=512):
    """
    Splits the provided text into chunks of a specified size.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

BATCH_SIZE = 32  # API limit

async def add_embeddings_to_collection(collection_name, text_data):
    """
    Sends text_data in batches (max 32 per request), retrieves embeddings, and adds them to ChromaDB.
    """
    collection = client.get_or_create_collection(name=collection_name)
    
    for i in range(0, len(text_data), BATCH_SIZE):
        batch = text_data[i:i + BATCH_SIZE]
        print("--")
        embeddings = await get_embeddings(batch)
        print("++")
        if embeddings:
            documents = [chunk.replace('\n', ' ').strip() for chunk in batch]
            ids = [f"{collection_name}_id_{i+j}" for j in range(len(documents))]

            collection.add(documents=documents, embeddings=embeddings, ids=ids)
            print(f"✅ Added {len(documents)} embeddings to collection: {collection_name}")
        else:
            print(f"⚠️ Failed to retrieve embeddings for batch {i // BATCH_SIZE + 1}. Skipping.")

async def process_pdf(pdf_path, collection_name):
    """
    Extracts text from a PDF, splits into chunks, and adds embeddings.
    Each chunk is prefixed with the page number.
    """
    if not os.path.exists(pdf_path):
        print(f"⚠️ File not found: {pdf_path}")
        return
    
    try:
        doc = fitz.open(pdf_path)
        all_chunks = []
        for page_num, page in enumerate(doc, start=1):  # Start page numbering from 1
            page_text = page.get_text("text")
            # Split each page's text into 512-token chunks
            chunks = await split_text_into_chunks(page_text, chunk_size=512)
            # Prepend the page number to each chunk
            chunks_with_page_num = [f"Page {page_num}: {chunk}" for chunk in chunks]
            all_chunks.extend(chunks_with_page_num)
        await add_embeddings_to_collection(collection_name, all_chunks)
    except Exception as e:
        print(f"⚠️ Error processing PDF: {e}")
        
async def query_database(collection_name, user_query, top_k=3):
    """
    Asynchronously queries the ChromaDB collection with the user's query, 
    converts the query to lowercase, and returns the top_k most relevant documents.
    """
    print(f"Querying database with top_k={top_k}")
    user_query_lower = user_query.lower()
    query_embedding_response = await get_embeddings([user_query_lower])
    
    if not query_embedding_response:
        print("⚠️ Failed to retrieve embedding for query.")
        return None

    query_embedding = query_embedding_response[0] if query_embedding_response else None
    if not query_embedding:
        print("⚠️ No embedding found in the response for the query.")
        return None
    
    collection = client.get_collection(name=collection_name)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    if results and 'documents' in results:
        return results['documents'][0]
    else:
        print("⚠️ No results found in the collection.")
        return None

async def get_client():
    """
    Provides the ChromaDB client instance asynchronously to be used in other parts of the application.
    """
    return client
