import pixellib
from pixellib.instance import custom_segmentation

test_video = custom_segmentation()
test_video.inferConfig(num_classes=  2, class_names=["BG", "butterfly", "squirrel"])
test_video.load_model("Nature_model_resnet101")
test_video.process_video("sample_video1.mp4", show_bboxes = True,  output_video_name="video_out.mp4", frames_per_second=15)