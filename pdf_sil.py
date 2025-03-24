from fastapi import APIRouter, Depends, HTTPException
from pathlib import Path
from vectordb import get_client
from pydantic import BaseModel

UPLOAD_DIR = Path("pdfler")

router = APIRouter()

# Pydantic model to define the expected JSON structure
class DeleteRequest(BaseModel):
    filename: str

@router.delete("/delete/")
async def delete_file(request: DeleteRequest, client=Depends(get_client)):
    # Validate the filename
    if not request.filename or request.filename.strip() == "" or "\n" in request.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not request.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail=" Dosya '.pdf' uzantılı olmalı")

    # Construct the file path
    file_path = UPLOAD_DIR / request.filename
    file_existed = file_path.exists()
    if file_existed:
        file_path.unlink()

    # Derive the collection name
    collection_name = request.filename.lower().replace(" ", "").replace(".pdf", "")
    existing_collections = client.list_collections()
    collection_existed = collection_name in {col.name for col in existing_collections}
    if collection_existed:
        client.delete_collection(collection_name)

    return {
        "filename": request.filename,
        "file_deleted": file_existed,
        "collection_deleted": collection_existed
    }