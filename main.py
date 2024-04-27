
from utils import (read_video, save_video, measure_distance, draw_player_stats, convert_meter_to_pixel_distance, convert_pixel_distance_to_meter)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2
from mini_court import MiniCourt
from copy import deepcopy
import pandas as pd
from constants import *

def main():
    
    #reading video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)
    
    #detecting players and ball
    Player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path="models/yolo5_last.pt")
    Player_detections = Player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    
    #Court line detector model
    Court_Line_Detector = CourtLineDetector(model_path="models/keypoints_model.pth")
    court_keypoints = Court_Line_Detector.predict(video_frames[0])
    
    # choose players
    Player_detections = Player_tracker.choose_and_filter_players(court_keypoints, Player_detections)
    
    #Draw MiniCourt
    mini_court = MiniCourt(video_frames[0])
    
    #Detect ball shots frames
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    print(ball_shot_frames)
    
    # convert the positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(Player_detections, ball_detections, court_keypoints)
    
    
    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    } ]
    
    
    for ball_shot_ind in range(len(ball_shot_frames) -1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24   #because it is 24 fps
        
        
        #Get distance covered by ball
        distance_covered_by_ball_in_pixels = measure_distance(ball_mini_court_detections[start_frame][1], ball_mini_court_detections[end_frame][1])
        
        distance_covered_by_ball_in_meters = convert_pixel_distance_to_meter(distance_covered_by_ball_in_pixels, DOUBLE_LINE_WIDTH, mini_court.get_width_of_mini_court())
        
        #Get speed of ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_in_meters / ball_shot_time_in_seconds * 3.6
        
        #Getting player who shot the ball
        player_positions = player_mini_court_detections[start_frame]
        
        player_shot_ball = min(player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id], ball_mini_court_detections[start_frame][1]))  #this will return player id with min distance
        
        
        #Get speed of opponent player
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_the_opponent_player_in_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id], player_mini_court_detections[end_frame][opponent_player_id])
        distance_covered_opponent_in_meters = convert_pixel_distance_to_meter(distance_covered_by_the_opponent_player_in_pixels, DOUBLE_LINE_WIDTH, mini_court.get_width_of_mini_court())
        
        #calculate speed of opponent player in km/h
        speed_of_opponent_player = distance_covered_opponent_in_meters / ball_shot_time_in_seconds * 3.6
        
        
        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot
        
        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent_player
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent_player
        
        player_stats_data.append(current_player_stats)
         
    player_stats_data_df = pd.DataFrame(player_stats_data)   
    frames_df = pd.DataFrame({'frame_num' : list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()
    
    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_1_number_of_shots']
    
    
    

    
                             
                             
    

    
    #draw output
    
    ## draw bboxes
    output_video_frames = Player_tracker.draw_bboxes(video_frames, Player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    
    
    #Drawing court keypoints
    output_video_frames = Court_Line_Detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    
    #Draw minicourt on top of frames
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0,255,255))
    
    #Drawing player stats
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    
    #Add frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        frame = cv2.putText(frame, f"Frame: {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        output_video_frames[i] = frame

    
    #saving video
    save_video(output_video_frames, "output_videos/output_video.avi")
    
    
    
    
    
    
if __name__ == "__main__":
    main()