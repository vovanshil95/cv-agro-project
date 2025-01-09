import cv2
import numpy as np
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
import torch.nn.functional as F
import pytorch_lightning as pl 
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import datasets
import torch
from torchmetrics import Metric
import torchmetrics
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import random
import os
from tqdm import tqdm
import math
from copy import deepcopy


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
cfg.MODEL.EVALUATOR_TYPE = 'coco'
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


predictor = DefaultPredictor(cfg)


def visualize_segments_tomats(image_folder):

    plt.figure(figsize=(24, 10))

    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg'))]

    sample_images = random.sample(image_files, min(8, len(image_files)))


    for i, image_file in enumerate(tqdm(sample_images)):

        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        outputs = predictor(image)

        v = Visualizer(image[:, :, ::-1], metadata=None, scale=1, instance_mode=ColorMode.IMAGE)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        rgb_image = out.get_image()[:, :, ::1]  

        plt.subplot(2, 4, i + 1)
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.title(f"{image_file}")

    plt.tight_layout()
    plt.show()


visualize_segments_tomats("data/Зрелый")
visualize_segments_tomats("data/Незрелый")
visualize_segments_tomats("data/В процесе созревания")