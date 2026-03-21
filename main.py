from utils.video_utils import read_video, save_video
from trackers import tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

def main():
    # read vid
    video_path = "D:\\Projects\\End-to-End\\Football-Analysis-System\\input_videos\\08fd33_4.mp4"
    video_frames = read_video(video_path)

    

    # init tracker
    tracker_instance = tracker.Trackers("models\\best.pt")
    tracks = tracker_instance.get_object_tracks(video_frames, 
                                                read_from_stub=True, 
                                                stub_path="stubs\\track_stubs.pkl")

    # interpolate ball positions
    tracks['ball'] = tracker_instance.interpolate_ball_position(tracks['ball'])

    # init team assigner
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                        tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, player_info in player_track.items():
            bbox = player_info['bbox']
            team_label = team_assigner.get_player_team(video_frames[frame_num], bbox, player_id)
            tracks['players'][frame_num][player_id]['team'] = team_label
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team_label]


    # Assign Ball Possesion
    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if(assigned_player!=-1):
            tracks["players"][frame_num][assigned_player]['has_ball']=True
            


    # draw output
    # draw object tracks on frames
    output_video_frames = tracker_instance.draw_annotations(video_frames, tracks)


    # save video
    output_video_path = "D:\\Projects\\End-to-End\\Football-Analysis-System\\output_videos\\output_video.avi"
    save_video(output_video_frames, output_video_path)

if __name__ == "__main__":
    main()