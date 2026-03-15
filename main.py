from utils.video_utils import read_video, save_video
from trackers import tracker
import cv2

def main():
    # read vid
    video_path = "D:\\Projects\\End-to-End\\Football-Analysis-System\\input_videos\\08fd33_4.mp4"
    video_frames = read_video(video_path)

    

    # init tracker
    tracker_instance = tracker.Trackers("models\\best.pt")
    tracks = tracker_instance.get_object_tracks(video_frames, 
                                                read_from_stub=True, 
                                                stub_path="stubs\\track_stubs.pkl")

    # save cropped image of a player
    for track_id, player in tracks["players"][0].items():
        bbox = player["bbox"]
        frame = video_frames[0]
        # print(bbox)

        # convert floats to ints and clamp within frame bounds
        x1 = int(round(bbox[0]))
        y1 = int(round(bbox[1]))
        x2 = int(round(bbox[2]))
        y2 = int(round(bbox[3]))

        # crop bbox of player from frame
        cropped_player = frame[y1:y2, x1:x2]

        # save cropped image
        cv2.imwrite(f"output_videos/cropped_player_{track_id}.jpg", cropped_player)

        break

    # draw output
    # draw object tracks on frames
    output_video_frames = tracker_instance.draw_annotations(video_frames, tracks)


    # save video
    output_video_path = "D:\\Projects\\End-to-End\\Football-Analysis-System\\output_videos\\output_video.avi"
    save_video(output_video_frames, output_video_path)

if __name__ == "__main__":
    main()