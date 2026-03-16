from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    
    def get_clustering_model(self, image):
        # reshape the image to 2d
        image_2d= image.reshape(-1, 3)

        # Implement your clustering model here (e.g., KMeans)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)  # Assuming two teams
        return kmeans

    def player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[:image.shape[0]//2, :]

        # get clustering model
        kmeans=self.get_clustering_model(top_half_image)
        # get the cluster labels for each pixel
        labels = kmeans.labels_
        # reshape the labels to image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        # get the player cluster
        corner_cluster = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)
        player_cluster = 1 - non_player_cluster
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame, player_detections):
        player_color = []
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_color.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means ++", n_init=1)
        kmeans.fit(player_color)
        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, bbox, player_id):
        if(player_id in self.player_team_dict):
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, bbox)
        team_label = self.kmeans.predict([player_color.reshape(1, -1)])[0]
        team_label += 1  # Convert to 1 and 2
        self.player_team_dict[player_id] = team_label  # Team labels are 1 and 2

        return team_label  # Team labels are 1 and 2