import sys
sys.path.append("..")
from utils import get_center_of_bbox

class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball_to_players(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        for player_id, player_info in players.items():
            player_bbox = player_info["bbox"]
            player_position = get_center_of_bbox(player_bbox)

            distance = ((ball_position[0] - player_position[0]) ** 2 + (ball_position[1] - player_position[1]) ** 2) ** 0.5

            if distance < self.max_player_ball_distance:
                players[player_id]["has_ball"] = True
            else:
                players[player_id]["has_ball"] = False