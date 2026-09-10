from fastapi import FastAPI
from app.routers import notes

app = FastAPI(
    title="AI Medical Scribe",
    description="Audio -> transcript -> diarized -> structured, grounded SOAP note",
    version="0.1.0",
)

app.include_router(notes.router, tags=["notes"])


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}
