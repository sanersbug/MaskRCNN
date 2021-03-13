import os
import pixellib
from pixellib.instance import custom_segmentation

# segment_image = custom_segmentation()
# segment_image.inferConfig(num_classes= 2, class_names= ["BG", "cl", "cz"])
# segment_image.load_model("mask_rcnn_models/mask_rcnn_model.124-0.027224.h5")
# segment_image.segmentImage("./image/00000002.tif", show_bboxes=True, output_image_name="sample_out.jpg")

image_path = './image/'
pre_path = './predict/'
imgs = os.listdir(image_path)
segment_image = custom_segmentation()
segment_image.inferConfig(num_classes= 2, class_names= ["BG", "cl", "cz"], network_backbone = "resnet50",detection_threshold = 0.5, image_max_dim = 512, image_min_dim = 512)
segment_image.load_model("mask_rcnn_models/mask_rcnn_model.124-0.027224.h5")
for im in imgs:
	name = im.split('.')[0]
	im_path = os.path.join(image_path, im)
	save_path = os.path.join(pre_path, name + '.png')
	segment_image.segmentImage(im_path, show_bboxes=True, output_image_name=save_path)