import json
from models.Layout_Model import LayoutModel
from models.Time_Stamp_Extractor import TimeStampExtractor
from models.Frame_Extractor import FrameExtractor
from models.Transcription_Model import WhisperTranscriber
from models.Transcript_Chunker import TranscriptChunker
from models.GPT_Model import GPTModel
from models.Video_Manager import VideoManager
import os
import shutil
import random


#### Program #### 

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "03_05_csp_local_search.mp4")

# create directories for data
data_output_dir = "data"
all_videos_output_dir = os.path.join(data_output_dir, "lecture_videos")         # stores all mp4 lecture videos
all_frames_output_dir = os.path.join(data_output_dir, "frames")             # stores all extracted jpg frames from the videos
all_layouts_output_dir = os.path.join(data_output_dir, "layouts")           # stores all layout data of the extracted frames (output of laytout detection model)
all_transcripts_output_dir = os.path.join(data_output_dir, "transcripts")       # stores the complete transcripts of the lecture videos with additional segment information

lecture_video_name = os.path.splitext(os.path.basename(video_path))[0]      # lecture video name without 'mp4'

videos_output_dir = os.path.join(all_videos_output_dir, lecture_video_name)
frames_output_dir = os.path.join(all_frames_output_dir, lecture_video_name)
layouts_output_dir = os.path.join(all_layouts_output_dir, lecture_video_name)
transcripts_output_dir = os.path.join(all_transcripts_output_dir, lecture_video_name)

# Now create the folders or check if existing already
os.makedirs(videos_output_dir, exist_ok=True)
os.makedirs(transcripts_output_dir, exist_ok=True)
os.makedirs(frames_output_dir, exist_ok=True)
os.makedirs(layouts_output_dir, exist_ok=True)

VideoManager.copy_video_to_data_dir(
    source_path=video_path, 
    video_name=lecture_video_name, 
    dest_dir=videos_output_dir
    )

VideoManager.store_metadata(
    video_path=video_path, 
    dest_dir=videos_output_dir
    )


# copy_video_to_data_dir(src_video_path=video_path, video_name=lecture_video_name, dest_dir=videos_output_dir)

# extract the time stamps (frame indices) at which the slides are changing
timeExtractor = TimeStampExtractor(video_path, sample_rate = 0.2)
slideChanges = timeExtractor.extract_timestamps_and_store(frames_output_dir)

# extract the frames at the previoiusly defined indices
frameExtractor = FrameExtractor(video_path, output_dir=frames_output_dir)
frameExtractor.get_frames_and_store(slideChanges)             # frame indices implicitly casted to miliseconds

# extract the complete transcript in the preprocessing step
transcriber = WhisperTranscriber(video_path, output_dir=transcripts_output_dir)
full_transcript = transcriber.transcribe_and_store()

# chunk the transcript for embedding purpose
chunker = TranscriptChunker(output_dir=transcripts_output_dir)
chunks = chunker.chunk_transcript_and_store(full_transcript["segments"], enrich_with_gpt=True)


# run layout detection model and store png + json results
layoutDetector = LayoutModel(input_dir=frameExtractor.img_output_dir, output_dir=layouts_output_dir)
layoutDetector.run_and_store_all_frames()