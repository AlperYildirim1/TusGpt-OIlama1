from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pathlib import Path

router = APIRouter()

UPLOAD_DIR = Path("pdfler")

@router.get("/list_files/")
async def list_files():
    # Check if the directory exists
    if not UPLOAD_DIR.exists():
        return JSONResponse(status_code=404, content={"message": "Directory not found!"})

    # List all PDF files in the directory
    files = [file.name for file in UPLOAD_DIR.iterdir() if file.is_file()]
    
    return {"files": files}
