import pixellib
from pixellib.instance import custom_segmentation
import cv2


capture = cv2.VideoCapture(0)

segment_camera = custom_segmentation()
segment_camera.inferConfig(num_classes=2, class_names=["BG", "butterfly", "squirrel"])
segment_camera.load_model("Nature_model_resnet101.h5")
segment_camera.process_camera(capture, frames_per_second= 10, output_video_name="output_video.mp4", show_frames= True,
frame_name= "frame")