import pixellib
from pixellib.custom_train import instance_custom_training


train_maskrcnn = instance_custom_training()
train_maskrcnn.modelConfig(network_backbone = "resnet50", num_classes= 2)
train_maskrcnn.load_dataset("jy2")
train_maskrcnn.evaluate_model("mask_rcnn_models/mask_rcnn_model.124-0.027224.h5")