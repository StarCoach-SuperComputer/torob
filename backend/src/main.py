from pathlib import Path
from fastapi import FastAPI
from repository import Repository

app = FastAPI()

# Data directory (relative to project root)
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

repo = Repository()

@app.on_event("startup")
async def startup_event():
    global repo
    if DATA_DIR.exists():
        repo = Repository.load_from_directory(DATA_DIR)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "shops_count": len(repo.shops),
        "products_count": len(repo.products),
        "offers_count": len(repo.offers),
    }
