import os
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from prompt_optimizer import optimize_prompt
from image_generator import generate_image, save_image

load_dotenv()

app = FastAPI(title="Diffusion Pipeline API")

OUTPUT_DIR = Path("/workspace/jiyong/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


class GenerateRequest(BaseModel):
    text: str


class GenerateResponse(BaseModel):
    original_input: str
    optimized_prompt: str
    image_filename: str


@app.get("/")
def root():
    return {"status": "running", "message": "Diffusion Pipeline API"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="텍스트를 입력해주세요")

    optimized = optimize_prompt(req.text)

    image_bytes = generate_image(optimized)

    filename = f"{uuid.uuid4().hex}.png"
    save_image(image_bytes, OUTPUT_DIR / filename)

    return GenerateResponse(
        original_input=req.text,
        optimized_prompt=optimized,
        image_filename=filename,
    )


@app.get("/image/{filename}")
def get_image(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")
    return FileResponse(path, media_type="image/png")
