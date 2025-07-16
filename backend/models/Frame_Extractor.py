import os
import cv2

class FrameExtractor:
    def __init__(self, video_path, output_dir="./frames"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.img_output_dir = os.path.join(output_dir, "images")
        os.makedirs(self.img_output_dir, exist_ok=True)
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def get_frames_and_store(self, frame_indices):
        for idx in frame_indices:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video at frame {idx}")
                continue

            # Convert frame index to milliseconds using fps
            timestamp_ms = (idx / self.fps) * 1000
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
            success, frame = cap.read()

            if success:
                frame_path = os.path.join(self.img_output_dir, f"{idx}_frame.png")
                cv2.imwrite(frame_path, frame)
            else:
                print(f"Warning: Failed to read frame at index {idx} (timestamp: {timestamp_ms} ms)")

            cap.release()

        
        return self.img_output_dir
