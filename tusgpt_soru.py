import asyncio
import fitz
import time
from openai import AsyncOpenAI
import re
from fastapi import APIRouter
import os
from ollama import AsyncClient

router = APIRouter()
  
async def pdf_to_chunks_with_fitz(pdf_path, chunk_size=1024, start_page=1, end_page=100):
    """
    Split PDF text into chunks from specified page range.
    """
    try:
        doc = fitz.open(pdf_path)
        chunks = []

        total_pages = len(doc)
        start_page = max(1, min(start_page, total_pages))
        end_page = max(start_page, min(end_page, total_pages))

        for i in range(start_page - 1, end_page):
            try:
                page_text = doc[i].get_text()
                
                # Handle empty pages
                if not page_text.strip():
                    continue
                    
                # Split into chunks with overlap to maintain context
                for j in range(0, len(page_text), chunk_size):
                    chunk = page_text[j:j + chunk_size]
                    if chunk.strip():  # Only add non-empty chunks
                        chunks.append({"chunk": chunk, "page": i + 1})
                        
            except Exception as e:
                print(f"Error processing page {i + 1}: {str(e)}")
                continue

        return chunks

    except Exception as e:
        print(f"Error opening PDF: {str(e)}")
        return []
    
async def get_response(chunks, chunk_start, chunk_end):
    try:
        # Validate chunk indices
        if not chunks:
            return "Error: No chunks available"
            
        chunk_start = max(0, chunk_start)
        chunk_end = min(chunk_end, len(chunks))
        
        if chunk_start >= chunk_end:
            return "Error: Invalid chunk range"

        selected_chunks = chunks[chunk_start:chunk_end]
        chunk_texts = []

        for i, chunk in enumerate(selected_chunks):
            chunk_text = await clean_chunk_text(chunk['chunk'])  # Assuming this is defined elsewhere
            page_number = chunk['page']
            formatted_chunk = f"Chunk {chunk_start + i} (Page {page_number}):\n{chunk_text}\n"
            chunk_texts.append(formatted_chunk)

        full_text = "\n".join(chunk_texts)    
        print(full_text)

        # Initialize Ollama async client
        client = AsyncClient(host="http://93.127.138.13:11434")  # Ollama's default local server
        
        start_time = time.time()
        
        # Call Ollama's chat API without streaming
        chat_completion = await client.chat(
            model="gemma3-12b-ctx1024:latest",  # Use your exact model tag if different
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sen doktorlar için bir asistansın. Görevin sınav soruları hazırlamak.\n"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Sorular 5 seçenek içermelidir (a, b, c, d, e) ve yalnızca bir tanesi doğru olmalıdır.\n"
                        f"Doğru cevap metinde yer almalıdır.\n"
                        f"JSON formatı şu alanları içermelidir: 'Page_number', 'Question', 'a', 'b', 'c', 'd', 'e', 'correct_answer' ve 'explanation'\n"
                        f"Bağlam dışı değilse en az 3 soru üretmeye çalış. Cevabın Json listesi şeklinde olduğundan emin ol.\n"
                        f"Son cevap sadece tek bir Json listesinden oluşmalı ve başka ek not veya cümle olmamalı. Json başlıkları belirtildiği gibi kalmalı ama içerikleri mutlaka Türkçe olmalı."
                        f"Ayrıca eğer yapabiliyorsan question kısmında bir tane hasta geçmişi oluşturup onun üzerinden bir soru hazırlayabilirsin."
                        f"Soruyu görecek olan kişi metni görmeyecek. Senin yapman gereken bu metindeki bilgileri kullanarak orijinal bir soru hazırlaman. Yani metne göre diye başlayarak soru sorma."
                        f"Eğer metni uygun görürsen sorulardan birini bir hasta geçmişi oluşturup soruyu bu hastanın durumu üzerinden zor bir soru olacak şekilde hazırla."
                        f"Metin:\n{full_text}\n"
                    )
                }
            ],
            options={
                "max_tokens": 1500,  # Ollama uses options dict for max_tokens
            },
            stream=False  # No streaming
        )
        
        # Return the full response content
        return chat_completion['message']['content']  # Ollama's response format

    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        return None

async def clean_chunk_text(text):
    if not isinstance(text, str):
        return ""
    return text.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
  
@router.get("/process_pdf")
async def process_pdf(pdf_path: str, start_page: int, end_page: int, chunk_size: int):
    # Generate chunks
    chunks = await pdf_to_chunks_with_fitz(
        pdf_path, 
        chunk_size=chunk_size, 
        start_page=start_page, 
        end_page=end_page
    )
    
    if not chunks:
        return {"error": "No chunks were generated. Check the PDF path and page range."}
    
    combined = ""
    total_chunks = len(chunks)

    # Process each chunk individually
    for i, chunk in enumerate(chunks):
        try:
            # Pass a single chunk to get_response
            response = await get_response([chunk], 0, 1)  # Wrap chunk in a list for consistency
            if response:
                combined += response + "\n"
                print(f"Processed chunk {i + 1}/{total_chunks}")
            else:
                raise Exception("Empty response")
        except Exception as e:
            print(f"Error processing chunk {i + 1}: {str(e)}")
        
        await asyncio.sleep(0.1)  # Delay between chunks
    
    print(combined)
    return {"final_response": combined, "length": len(combined)}