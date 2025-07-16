import json


def get_box_coordinates(json_input_file: str, box_id: int):
    with open (json_input_file, "r") as f:
        layoutJson = json.load(f)
    
    for box in layoutJson.get("boxes", []):
        if box.get("box_id") == box_id:
            return box['coordinate']
    
    return None  # if not found
