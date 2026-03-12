from utils.video_utils import read_video, save_video
from trackers import tracker

def main():
    # read vid
    video_path = "D:\\Projects\\End-to-End\\Football-Analysis-System\\input_videos\\08fd33_4.mp4"
    video_frames = read_video(video_path)

    # init tracker
    tracker_instance = tracker.Trackers("models\\best.pt")
    tracks = tracker_instance.get_object_tracks(video_frames, 
                                                read_from_stub=True, 
                                                stub_path="stubs\\track_stubs.pkl")

    # draw output
    # draw object tracks on frames
    output_video_frames = tracker_instance.draw_annotations(video_frames, tracks)


    # save video
    output_video_path = "D:\\Projects\\End-to-End\\Football-Analysis-System\\output_videos\\output_video.avi"
    save_video(output_video_frames, output_video_path)

if __name__ == "__main__":
    main()