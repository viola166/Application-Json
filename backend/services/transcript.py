import json
import os
from typing import List, Dict

# get the current transcript chunk, the five previous ones, and the 5 next ones
def get_transcript_chunks_for_pause(chunks_path: str, timestamp: float) -> List[Dict]:
    """
    Returns up to 9 chunks: 4 before, the current, and 4 after the chunk containing the timestamp.
    """
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Find the index of the current chunk
    current_index = next(
        (i for i, chunk in enumerate(chunks) if chunk["start"] <= timestamp < chunk["end"]),
        None
    )

    if current_index is None:
        return []

    # Calculate start and end indices, ensuring they stay within bounds
    start_index = max(0, current_index - 4)
    end_index = min(len(chunks), current_index + 5)  # +5 to include current + 4 after

    return " ".join(chunk["text"] for chunk in chunks[start_index:end_index])

