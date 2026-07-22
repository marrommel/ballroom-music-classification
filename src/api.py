from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, shutil
from src.config import Config
from src.helpers.inference import load_model, extract_chunks, predict

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

config = Config()
model = load_model(config.inference_model_weights)  # loaded once at startup

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        chunks = extract_chunks(tmp_path, config.spec_types)
        predicted_class, probabilities = predict(model, chunks)
        return {"predicted_class": predicted_class, "probabilities": probabilities}
    finally:
        os.unlink(tmp_path)

@app.get("/health")
def health(): return {"status": "ok"}