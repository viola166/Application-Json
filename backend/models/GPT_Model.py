import numpy as np
from openai import OpenAI
from typing import List, Union
from pathlib import Path
import base64

class GPTModel:
    _instance = None

    def __init__(self):
        self.client = OpenAI()  # Uses OPENAI_API_KEY from env

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [e.embedding for e in response.data]


    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    

    def label_chunk(self, chunk_text: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Label this transcript chunk concisely."},
                {"role": "user", "content": chunk_text}
            ]
        )
        return response.choices[0].message.content.strip()

    def _encode_image(self, image: Union[str, Path, bytes]) -> str:
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                image_data = f.read()
        elif isinstance(image, bytes):
            image_data = image
        else:
            raise ValueError("Image must be a file path or bytes.")

        return base64.b64encode(image_data).decode("utf-8")


    def explain(self, transcript: str, cropped_image: Union[str, Path, bytes], full_slide_image: Union[str, Path, bytes] = None) -> str:
        cropped_b64 = self._encode_image(cropped_image)
        images = [{
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{cropped_b64}"}
        }]

        if full_slide_image:
            slide_b64 = self._encode_image(full_slide_image)
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{slide_b64}"}
            })

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise tutor AI. "
                    "Explain the selected region clearly in 1–3 sentences, using the transcript for context. "
                    "Use the full slide also for context."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Transcript:\n{transcript}"},
                    {"type": "text", "text": "Cropped region:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{cropped_b64}"}}
                ] + (
                    [
                        {"type": "text", "text": "Full slide:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{slide_b64}"}}
                    ] if full_slide_image else []
                )
            }
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.4
        )

        return response.choices[0].message.content.strip()
