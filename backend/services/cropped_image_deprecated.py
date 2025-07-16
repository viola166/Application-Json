import json
import os
from pathlib import Path
from PIL import Image


def crop_box_from_image(image_path: str, box_coordinates: list) -> str:
    """
    Crops a box from an image using coordinates specified in a JSON layout file.

    Args:
        image_path (str): Path to the frame image.
        json_path (str): Path to the JSON file with box coordinates.
        box_index (int): Index of the box to crop.
        output_dir (str): Directory to save the cropped image.

    Returns:
        Image: the cropped image defined by the box coordinates
    """
    image = Image.open(image_path).convert("RGB")

    # with open(json_path, "r", encoding="utf-8") as f:
    #     layout_data = json.load(f)

    # boxes = layout_data.get("boxes", [])

    # if box_id < 0 or box_id >= len(boxes):
    #     raise IndexError(f"Box index {box_id} out of range. Total boxes: {len(boxes)}")

    # # Find the box with the matching box_id
    # box = next((box for box in boxes if box.get("box_id") == box_id), None)
    # if box is None:
    #     raise ValueError(f"Box with box_id {box_id} not found in layout.")

    x1, y1, x2, y2 = box_coordinates
    cropped = image.crop((x1, y1, x2, y2))


    # os.makedirs(output_dir, exist_ok=True)
    # filename = f"{Path(image_path).stem}_box{box_id}.png"
    # output_path = os.path.join(output_dir, filename)
    # cropped.save(output_path)

    return cropped
