from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from repository import Repository
from controller import home_router

# Load .env file from backend root or current directory
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

app = FastAPI(title="Torob Search & RAG API")

# Data directory (relative to project root)
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

repo = Repository()

@app.on_event("startup")
async def startup_event():
    global repo
    if DATA_DIR.exists():
        repo = Repository.load_from_directory(DATA_DIR)
    app.state.repo = repo

# Mount controller router
app.include_router(home_router)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "shops_count": len(repo.shops),
        "products_count": len(repo.products),
        "offers_count": len(repo.offers),
    }
