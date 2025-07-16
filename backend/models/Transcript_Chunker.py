import json
import os
from openai import OpenAI
import numpy as np
from typing import List, Dict, Optional
from GPT_Model import GPTModel

class TranscriptChunker:
    def __init__(self, embed_model: str = "text-embedding-3-small", similarity_threshold: float = 0.26, output_dir=".\transcripts"):
        self.embed_model = embed_model
        self.similarity_threshold = similarity_threshold
        self.client = GPTModel.get_instance()  # Create OpenAI client
        self.output_dir = output_dir


    def chunk_transcript_and_store(self, transcript_segments: List[Dict], enrich_with_gpt: bool = False) -> List[Dict]:
        """
        Takes a list of Whisper-style transcript segments and returns semantic chunks.
        Each segment should have: {"start": float, "end": float, "text": str}
        """
        if not transcript_segments:
            return []

        texts = [seg["text"] for seg in transcript_segments]
        embeddings = self.client.get_embeddings(texts)

        chunks = []
        current_chunk = {
            "start": transcript_segments[0]["start"],
            "end": transcript_segments[0]["end"],
            "text": transcript_segments[0]["text"],
        }

        for i in range(1, len(transcript_segments)):
            sim = self.client.cosine_sim(embeddings[i], embeddings[i - 1])

            # if the similarity between the two segments is lower than the threshold,
            # then they are not chunked together and a new chunk begins
            if sim < self.similarity_threshold:
                if enrich_with_gpt:
                    current_chunk["label"] = self.client.label_chunk(current_chunk["text"])
               
                chunk_embedding = self.client.get_embeddings([current_chunk["text"]])[0]
                current_chunk["embedding"] = chunk_embedding
                chunks.append(current_chunk)
                current_chunk = {
                    "start": transcript_segments[i]["start"],
                    "end": transcript_segments[i]["end"],
                    "text": transcript_segments[i]["text"],
                }
            else:
                current_chunk["end"] = transcript_segments[i]["end"]
                current_chunk["text"] += " " + transcript_segments[i]["text"]

        # Add final chunk
        if enrich_with_gpt:
            current_chunk["label"] = self.client.label_chunk(current_chunk["text"])
        
        chunk_embedding = self.client.get_embeddings([current_chunk["text"]])[0]
        current_chunk["embedding"] = chunk_embedding
        chunks.append(current_chunk)

        output_path = os.path.join(self.output_dir, "chunks.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        return chunks
