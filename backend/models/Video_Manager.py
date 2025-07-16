import shutil
import os
import cv2
import json

class VideoManager:
    @staticmethod
    def copy_video_to_data_dir(source_path: str, video_name: str, dest_dir: str):
        dest_path = os.path.join(dest_dir, video_name)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(source_path, dest_path)
        return dest_path

    @staticmethod
    def store_metadata(video_path: str, dest_dir: str):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        metadata = {
            "fps": fps,
            "width": width,
            "height": height
        }

        os.makedirs(dest_dir, exist_ok=True)

        with open(os.path.join(dest_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        cap.release()