from utils.video_utils import read_video, save_video
from trackers import tracker

def main():
    # read vid
    video_path = "D:\\Projects\\End-to-End\\football-analysis-cv\\input_videos\\08fd33_4.mp4"
    frames = read_video(video_path)

    # save video
    output_video_path = "D:\\Projects\\End-to-End\\football-analysis-cv\\output_videos\\output_video.avi"
    save_video(frames, output_video_path)

if __name__ == "__main__":
    main()