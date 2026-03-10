from utils.video_utils import read_video, save_video
from trackers import tracker

def main():
    # read vid
    video_path = "D:\\Projects\\End-to-End\\Football-Analysis-System\\input_videos\\08fd33_4.mp4"
    frames = read_video(video_path)

    # init tracker
    tracker_instance = tracker.Trackers("models/best.pt")
    tracks = tracker_instance.get_object_tracks(frames)

    # save video
    output_video_path = "D:\\Projects\\End-to-End\\Football-Analysis-System\\output_videos\\output_video.avi"
    save_video(frames, output_video_path)

if __name__ == "__main__":
    main()