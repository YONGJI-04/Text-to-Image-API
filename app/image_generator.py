import os
import io
import base64
import requests
from PIL import Image


HF_API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"


def generate_image(prompt: str) -> bytes:
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "width": 1024,
            "height": 1024,
        },
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f"HuggingFace API error {response.status_code}: {response.text}")

    return response.content


def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def save_image(image_bytes: bytes, path: str) -> None:
    image = Image.open(io.BytesIO(image_bytes))
    image.save(path)
