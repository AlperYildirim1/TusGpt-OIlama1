from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
import shutil
from pathlib import Path
import asyncio
from vectordb import get_client  # Import ChromaDB client
from vectordb import process_pdf

UPLOAD_DIR = Path("pdfler")

router = APIRouter()

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...), client=Depends(get_client)):
    file_path = UPLOAD_DIR / file.filename
    collection_name = file.filename.lower().replace(" ", "").replace(".pdf", "")

    if file_path.exists():
        raise HTTPException(status_code=400, detail="Bu isimde bir PDF zaten yüklü")

    existing_collections = client.list_collections()
    if collection_name in {col.name for col in existing_collections}:
        raise HTTPException(status_code=400, detail="Koleksiyon zaten mevcut, dosya ismini değiştirmeyi deneyin")

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    await process_pdf(file_path, collection_name)

    return {"filename": file.filename, "collection": collection_name, "status": "uploaded"}
