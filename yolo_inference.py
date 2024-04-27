from ultralytics import YOLO
#model = YOLO('yolov8x')
#result = model.predict('input_videos/input_video.mp4', save=True)

# using trained model to predict the video 

model = YOLO('yolov8x')
result = model.track('input_videos/input_video.mp4',conf=0.2, save=True)
print(result)