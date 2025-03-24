from vectordb import process_pdf
import asyncio
import os

async def main():
    # Update to the new filename
    file_path = "/mnt/pdfs/Robin.pdf"
    print(f"Attempting to process file at: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    await process_pdf(file_path, "Harrison")

# Run the main function within an event loop
if __name__ == "__main__":
    asyncio.run(main())