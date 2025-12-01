import os
import pickle
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from reader3 import Book, BookMetadata, ChapterContent, TOCEntry
import google.generativeai as genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")

class ChatRequest(BaseModel):
    book_id: str
    chapter_index: int
    message: str
    history: list = Field(default_factory=list)


app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Where are the book folders located?
BOOKS_DIR = "."

@lru_cache(maxsize=10)
def load_book_cached(folder_name: str) -> Optional[Book]:
    """
    Loads the book from the pickle file.
    Cached so we don't re-read the disk on every click.
    """
    file_path = os.path.join(BOOKS_DIR, folder_name, "book.pkl")
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "rb") as f:
            book = pickle.load(f)
        return book
    except Exception as e:
        print(f"Error loading book {folder_name}: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def library_view(request: Request):
    """Lists all available processed books."""
    books = []

    # Scan directory for folders ending in '_data' that have a book.pkl
    if os.path.exists(BOOKS_DIR):
        for item in os.listdir(BOOKS_DIR):
            if item.endswith("_data") and os.path.isdir(item):
                # Try to load it to get the title
                book = load_book_cached(item)
                if book:
                    books.append({
                        "id": item,
                        "title": book.metadata.title,
                        "author": ", ".join(book.metadata.authors),
                        "chapters": len(book.spine)
                    })

    return templates.TemplateResponse("library.html", {"request": request, "books": books})

@app.get("/read/{book_id}", response_class=HTMLResponse)
async def redirect_to_first_chapter(book_id: str):
    """Helper to just go to chapter 0."""
    return await read_chapter(book_id=book_id, chapter_index=0)

@app.get("/read/{book_id}/{chapter_index}", response_class=HTMLResponse)
async def read_chapter(request: Request, book_id: str, chapter_index: int):
    """The main reader interface."""
    book = load_book_cached(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    if chapter_index < 0 or chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")

    current_chapter = book.spine[chapter_index]

    # Calculate Prev/Next links
    prev_idx = chapter_index - 1 if chapter_index > 0 else None
    next_idx = chapter_index + 1 if chapter_index < len(book.spine) - 1 else None

    return templates.TemplateResponse("reader.html", {
        "request": request,
        "book": book,
        "current_chapter": current_chapter,
        "chapter_index": chapter_index,
        "book_id": book_id,
        "prev_idx": prev_idx,
        "next_idx": next_idx,
        "has_gemini": bool(GEMINI_API_KEY),
    })

@app.get("/read/{book_id}/images/{image_name}")
async def serve_image(book_id: str, image_name: str):
    """
    Serves images specifically for a book.
    The HTML contains <img src="images/pic.jpg">.
    The browser resolves this to /read/{book_id}/images/pic.jpg.
    """
    # Security check: ensure book_id is clean
    safe_book_id = os.path.basename(book_id)
    safe_image_name = os.path.basename(image_name)

    img_path = os.path.join(BOOKS_DIR, safe_book_id, "images", safe_image_name)

    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(img_path)

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")

    book = load_book_cached(request.book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    if request.chapter_index < 0 or request.chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")

    chapter = book.spine[request.chapter_index]
    context_text = chapter.text[:30000]

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # System prompt / Context
        system_instruction = f"You are a helpful assistant helping a user read a book. The user is currently reading a chapter with the following content:\n\n{context_text}\n\nAnswer the user's questions based on this text in plain text."
        
        full_prompt = f"{system_instruction}\n\nUser: {request.message}"
        
        response = model.generate_content(full_prompt)
        
        return {"response": response.text}
    except Exception as e:
        print(f"Gemini API Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

@app.post("/api/summarize")
async def summarize_endpoint(request: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")

    book = load_book_cached(request.book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    chapter = book.spine[request.chapter_index]
    context_text = chapter.text[:30000]

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = (
            "Provide a richer summary (about 5-7 sentences) in plain text of the chapter below. "
            "Include key insights, events, important names, and any notable details. "
            "Keep it concise but not terse.\n\n"
            f"{context_text}"
        )
        
        response = model.generate_content(prompt)
        return {"summary": response.text}
    except Exception as e:
        print(f"Gemini API Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://127.0.0.1:8124")
    uvicorn.run(app, host="127.0.0.1", port=8124)
