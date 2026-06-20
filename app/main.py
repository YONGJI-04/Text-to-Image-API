import os
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv

from prompt_optimizer import optimize_prompt
from image_generator import generate_image, save_image

load_dotenv()

app = FastAPI(
    title="Text to Image API",
    description="한국어/영어 텍스트를 입력하면 Claude API가 프롬프트를 최적화하고 FLUX.1이 이미지를 생성합니다.",
    version="1.1.0",
)

OUTPUT_DIR = Path("/workspace/jiyong/claude-flux-pipeline/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SIZE_MAP = {
    "small":  (512, 512),
    "medium": (768, 768),
    "large":  (1024, 1024),
}

class GenerateRequest(BaseModel):
    text: str
    size: Literal["small", "medium", "large"] = "large"

class GenerateResponse(BaseModel):
    original_input: str
    optimized_prompt: str
    image_filename: str
    size: str

@app.get("/")
def root():
    return {"status": "running", "message": "Text to Image API - Claude + FLUX.1"}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="텍스트를 입력해주세요")

    optimized = optimize_prompt(req.text)
    width, height = SIZE_MAP[req.size]
    image_bytes = generate_image(optimized, width=width, height=height)

    filename = f"{uuid.uuid4().hex}.png"
    save_image(image_bytes, OUTPUT_DIR / filename)

    return GenerateResponse(
        original_input=req.text,
        optimized_prompt=optimized,
        image_filename=filename,
        size=req.size,
    )

@app.get("/image/{filename}")
def get_image(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")
    return FileResponse(path, media_type="image/png")

@app.get("/sizes")
def get_sizes():
    return {"sizes": {k: {"width": v[0], "height": v[1]} for k, v in SIZE_MAP.items()}}
