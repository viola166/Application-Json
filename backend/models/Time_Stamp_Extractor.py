import cv2
import os
import json

# detecting time stamps of slide changes by comparing two neighbor frames each with their pixel-wise absolute difference
# lightweight approach: reducing image dimensions to 100x100
class TimeStampExtractor:

    def __init__(self, video_path, sample_rate=0.2, diff_threshold=2, resize_dim=(100, 100)):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.video_path = video_path
        self.sample_rate = sample_rate  # frames per second to sample; default: every 5 seconds
        self.diff_threshold = diff_threshold
        self.resize_dim = resize_dim

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_interval = int(self.fps / self.sample_rate) if self.sample_rate > 0 else 1

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

    def _process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.resize_dim)
        return resized    


    def extract_timestamps_and_store(self, output_dir):
        ret, prev_frame = self.cap.read()
        if not ret:
            raise ValueError("Cannot read video")

        prev_processed = self._process_frame(prev_frame)
        slide_change_frames = [0]  # first frame is a slide start
        frame_idx = 1

        while True:
            # Skip frames to sample at correct interval
            for _ in range(self.frame_interval - 1):
                if not self.cap.grab():
                    self.cap.release()

                    # store list as json file
                    os.makedirs(output_dir, exist_ok=True)
                    with open(os.path.join(output_dir, "frame_indices.json"), "w") as f:
                        json.dump(slide_change_frames, f)
                    
                    return slide_change_frames

            ret, frame = self.cap.read()
            if not ret:
                break

            curr_processed = self._process_frame(frame)
            diff = cv2.absdiff(curr_processed, prev_processed)
            diff_mean = diff.mean()

            if diff_mean > self.diff_threshold:
                slide_change_frames.append(frame_idx)

            prev_processed = curr_processed
            frame_idx += self.frame_interval

        self.cap.release()

        # store list as json file
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "frame_indices.json"), "w") as f:
            json.dump(slide_change_frames, f)
        
        return slide_change_frames
    
