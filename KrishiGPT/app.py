"""
app.py
------
FastAPI entry point for KrishiGPT.

Serves two capabilities:
  1. /ask           -- multilingual (Marathi + English) RAG crop advisory chat.
  2. /predict-image -- crop-disease classification from an uploaded leaf photo.

The RAG side detects the farmer's language, retrieves in English, and answers
back in the original language. The vision side loads the trained PyTorch
transfer-learning model on demand.

Run:
    uvicorn app:app --reload
    # then open http://127.0.0.1:8000
"""

from fastapi import FastAPI, Form, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from chat1 import fetch_website_content, extract_pdf_text, initialize_vector_store
from chat2 import setup_retrieval_qa
from rag.multilingual import detect_language, translate, language_name

app = FastAPI(title="KrishiGPT")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------------------------
# Build the knowledge base once at startup.
# Each source is loaded defensively so a blocked URL or missing PDF degrades
# gracefully instead of taking down the whole server.
# ---------------------------------------------------------------------------
urls = ["https://mospi.gov.in/4-agricultural-statistics"]
pdf_files = ["Data/Farming Schemes.pdf", "Data/farmerbook.pdf"]

documents = []
for url in urls:
    try:
        documents.append(fetch_website_content(url))
    except Exception as exc:  # network blocked / timeout
        print(f"[KrishiGPT] Skipping URL {url}: {exc}")
for pdf in pdf_files:
    try:
        documents.append(extract_pdf_text(pdf))
    except Exception as exc:  # missing / unreadable PDF
        print(f"[KrishiGPT] Skipping PDF {pdf}: {exc}")

print(f"[KrishiGPT] Building vector store from {len(documents)} source(s)...")
db = initialize_vector_store(documents)
print("[KrishiGPT] Knowledge base ready.")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Starlette >=0.29 expects `request` as the first positional argument.
    return templates.TemplateResponse(request, "index.html")


@app.post("/ask")
async def ask(messageText: str = Form(...)):
    query = messageText.strip()

    if query.lower() in ["who developed you?", "who created you?", "who made you?"]:
        return {"answer": "I was developed by Jayesh Bhandarkar."}

    # 1. Detect language and 2. translate Marathi -> English for retrieval.
    lang = detect_language(query)
    english_query = translate(query, source=lang, target="en")

    # 3. Answer in the farmer's original language.
    chain = setup_retrieval_qa(db, answer_language=language_name(lang))
    response = chain({"query": english_query})
    return {"answer": response["result"], "language": language_name(lang)}


@app.post("/predict-image")
async def predict_image_route(image: UploadFile = File(...)):
    """Classify an uploaded crop-leaf image using the trained CNN."""
    # Import lazily so the chatbot still runs even before a model is trained.
    from vision.predict import predict_image

    try:
        results = predict_image(image.file, top_k=3)
    except FileNotFoundError:
        return JSONResponse(
            status_code=503,
            content={"error": "No trained model found. Run: python -m vision.train"},
        )

    return {"predictions": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
