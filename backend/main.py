import io
import math
from typing import List
import numpy as np
from pydantic import BaseModel
from PIL import Image
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import json


from .services.transcript import get_transcript_chunks_for_pause
from .services.image_transform import pil_image_to_bytes
from .services.gpt import get_gpt_explanation, get_gpt_embedding, cosine_sim
from .services.coordinates import get_box_coordinates

app = FastAPI()

origins = [
    "http://localhost:5173",  # frontend URL and port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FRAME_DIR = "./backend/data/frames"
# LAYOUT_DIR = "./backend/data/layouts"
# VIDEO_DIR = "./backend/data/lecture_videos"

# Use absolute paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

FRAME_DIR = os.path.join(DATA_DIR, "frames")
LAYOUT_DIR = os.path.join(DATA_DIR, "layouts")
VIDEO_DIR = os.path.join(DATA_DIR, "lecture_videos")
TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcripts")

# Serve the entire 'data' folder at /data URL prefix

app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")




# "/video/{video_name}" is the endpoint that comes with a communication exchange when it's active
# here: GET request: recieving information from that endpoint
@app.get("/video/{video_name}")
def get_video(video_name: str):
    base_path = os.path.join(VIDEO_DIR, video_name)
    file_path = os.path.join(base_path, video_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Video not found")
    # whenever information is returned from an endpoint, Fast-API translates it into JSON format
    # whenever information is send to an endpoint, Fast-API translates JSON to vanilla python types
    return FileResponse(file_path, media_type="video/mp4")

# if the parameter in the function would not be given in the @app.get(parameter) line, then by 
# default this parameter is a "query parameter"
# query parameter in endpoint "?parameter=value" at the end of the path
@app.get("/layout/{video_name}/{frame_index}")
def get_layout_data(video_name: str, frame_index: str):
    base_path = os.path.join(LAYOUT_DIR, video_name)
    # json_path = os.path.join(base_path, "res", f"{frame_index}_frame.json")
    # img_path = os.path.join(base_path, "images", f"{frame_index}_frame.png")
    
    json_file = f"{frame_index}_frame.json"
    # img_file = f"{frame_index}_frame.png"

    json_path = os.path.join(base_path, "res", json_file)
    # img_path = os.path.join(base_path, "images", img_file)

    if not os.path.isfile(json_path):
        raise HTTPException(status_code=404, detail="Layout data not found")

    with open (json_path, "r") as f:
        return json.load(f)
    
    # return {
    #     "json": f"/data/layouts/{video_name}/res/{json_file}",
    #     "image": f"/data/layouts/{video_name}/images/{img_file}",
    # }

@app.get("/metadata/{video_name}")
def get_metadata(video_name: str):
    base_path = os.path.join(VIDEO_DIR, video_name)
    file_path = os.path.join(base_path, "metadata.json")

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Metadata of video not found")
    
    with open(file_path, "r") as f:
        return json.load(f)         # returns Python dict
    

@app.get("/frame/{video_name}/indices")
def get_available_frames(video_name: str):
    file_path = os.path.join(FRAME_DIR, video_name, "frame_indices.json")
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Frame indices data not found")
    
    with open(file_path, "r") as f:
        return json.load(f)         # list of integers
    

class ExplainRequest(BaseModel):
    video_name: str
    timestamp: float
    box_id: int


@app.post("/explain")
async def explain(request: ExplainRequest):

    video_name = request.video_name
    timestamp = request.timestamp
    box_id = request.box_id

    # === 1. Get transcript (placeholder logic) ===
    chunks_path = os.path.join(TRANSCRIPT_DIR, video_name, "chunks.json")
    transcript = get_transcript_chunks_for_pause(chunks_path, timestamp)

    # === 2. Get image and crop the image box ===
    # first find the right image    
    frame_indices = get_available_frames(video_name)
    video_fps = get_metadata(video_name)['fps']
    current_frame_index = math.floor(timestamp * video_fps)

    selected_frame = None
    for frame in frame_indices:
        if frame >= current_frame_index:
            selected_frame = frame
            break

    frame_img_path = os.path.join(FRAME_DIR, video_name, "images", f"{selected_frame}_frame.png")
    
    # retrieve the original box coordinates
    layout_json_path = os.path.join(LAYOUT_DIR, video_name, "res", f"{selected_frame}_frame.json")
    box_coordinates = get_box_coordinates(layout_json_path, box_id)

    if not box_coordinates:
        # Handle missing or empty coordinates
        raise ValueError(f"Coordinates missing for box id {box_id}")

    image = Image.open(frame_img_path).convert("RGB")
    x1, y1, x2, y2 = box_coordinates
    cropped_box_image = image.crop((x1, y1, x2, y2))


    # === 3. Get GPT-4o explanation (replace with your GPT handler) ===
    # bring the images into suitable format
    image_bytes = pil_image_to_bytes(image)
    cropped_image_bytes = pil_image_to_bytes(cropped_box_image)
    explanation = get_gpt_explanation(transcript=transcript, cropped_image=cropped_image_bytes, full_slide_image=image_bytes)

    return {"explanation": explanation}


class AssociateRequest(BaseModel):
    video_name: str
    timestamp: float
    explanation: str


@app.post("/associate")
async def associate_content(request: AssociateRequest):
    
    video_name = request.video_name
    timestamp = request.timestamp
    explanation = request.explanation

    explanation_embedding = get_gpt_embedding(explanation)

    # Load chunks
    chunks_path = os.path.join(TRANSCRIPT_DIR, video_name, "chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    past_chunks = [chunk for chunk in chunks if chunk["start"] <= timestamp]
    if not past_chunks:
        return {"error": "No prior chunks to compare with."}
    
    # making sure that the user is not directly navigated to the section right before
    filtered_chunks = past_chunks[:-4] if len(past_chunks) > 4 else []

    # Match
    best_sim = -1
    best_chunk = None
    for chunk in filtered_chunks:
        sim = cosine_sim(np.array(explanation_embedding), np.array(chunk["embedding"]))
        if sim > best_sim:
            best_sim = sim
            best_chunk = chunk
    
    return {
        "start": best_chunk["start"],
        "label": best_chunk.get("label"),
        "similarity": best_sim,
    } if best_chunk else {"error": "No matching chunk found."}








