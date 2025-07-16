from pathlib import Path
from typing import List, Union

import numpy as np
from ..models.GPT_Model import GPTModel

def get_gpt_explanation(transcript: str, cropped_image: Union[str, Path, bytes], full_slide_image: Union[str, Path, bytes]):
    gpt = GPTModel.get_instance()  # instantiate globally or inside your method
    explanation = gpt.explain(transcript=transcript, cropped_image=cropped_image, full_slide_image=full_slide_image)

    return explanation
    
def get_gpt_embedding(text: str) -> List[List[float]]:
    gpt = GPTModel.get_instance()
    embedding = gpt.get_embeddings([text])[0]
    return embedding

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
