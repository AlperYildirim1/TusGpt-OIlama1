from fastapi import FastAPI
import RAG_router
import pdf_yukle
import tusgpt_soru
import pdf_sil
import pdf_listele
app = FastAPI()

app.include_router(pdf_yukle.router, prefix="/api")
app.include_router(RAG_router.router)
app.include_router(tusgpt_soru.router)
app.include_router(pdf_sil.router)
app.include_router(pdf_listele.router)

@app.get("/")
def root():
    return {"message": "Welcome to FastAPI"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
