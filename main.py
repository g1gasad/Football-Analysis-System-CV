from utils.video_utils import read_video, save_video
from trackers import tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
def main():
    # read vid
    video_path = "D:\\Projects\\End-to-End\\Football-Analysis-System\\input_videos\\08fd33_4.mp4"
    video_frames = read_video(video_path)

    # init tracker
    tracker_instance = tracker.Trackers("models\\best.pt")
    tracks = tracker_instance.get_object_tracks(video_frames, 
                                                read_from_stub=True, 
                                                stub_path="stubs\\track_stubs.pkl")

    # Get object positions
    tracker_instance.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, 
                                            read_from_stub=True, stub_path="stubs/camera_movement_stub.pkl")

    tracks = camera_movement_estimator.adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # interpolate ball positions
    tracks['ball'] = tracker_instance.interpolate_ball_position(tracks['ball'])

    # init team assigner
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, player_info in player_track.items():
            bbox = player_info['bbox']
            team_label = team_assigner.get_player_team(video_frames[frame_num], bbox, player_id)
            tracks['players'][frame_num][player_id]['team'] = team_label
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team_label]


    # Assign Ball Possesion
    player_assigner = PlayerBallAssigner()
    team_ball_control=[]
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if(assigned_player!=-1):
            tracks["players"][frame_num][assigned_player]['has_ball']=True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # draw output
    # draw object tracks on frames
    output_video_frames = tracker_instance.draw_annotations(video_frames, tracks, team_ball_control)

    # draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # save video
    output_video_path = "D:\\Projects\\End-to-End\\Football-Analysis-System\\output_videos\\output_video.avi"
    save_video(output_video_frames, output_video_path)

if __name__ == "__main__":
    main()