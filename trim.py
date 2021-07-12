# CODE TO TRIM VIDEO
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
ffmpeg_extract_subclip("video1.mp4", start_time, end_time, targetname="test.mp4")