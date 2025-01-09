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



def calculate_angles(x, y, L1, L2, L3, L4):
    theta1 = math.acos(((x + L4)**2 + (y - L3)**2 + L1**2 - L2**2) / (2 * L1 * math.sqrt((x + L4)**2 + (y - L3)**2))) \
             + math.atan2(y - L3, x + L4)
    theta2 = math.acos((-((x + L4)**2 + (y - L3)**2) - L1**2 + L2**2) / (2 * L1 * L2))
    theta3 = math.pi - theta1 + theta2
    theta4 = math.pi / 2
    return {'theta1': theta1, 'theta2': theta2, 'theta3': theta3, 'theta4': theta4}


def get_object_centers_and_corners(image_path):
    image = cv2.imread(image_path)
    
    outputs = predictor(image)
    
    instances = outputs["instances"]
    boxes = instances.pred_boxes.tensor.cpu().numpy()
    
    centers = []
    corners = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        centers.append(({'x': float(center_x), 'y': float(center_y)}))
        
        corners.append(({'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)}))

    return {'centers': centers, 'corners': corners}



def get_angles_from_image(image_path, L1, L2, L3, L4):
    obj_coords = get_object_centers_and_corners(image_path)
    angles =  [calculate_angles(center['x'], center['y'], L1, L2, L3, L4) for center in obj_coords['centers']]
    coords = deepcopy(obj_coords)

    coords['angles'] = angles

    return coords


image_path = 'data/Незрелый/незрелые (30).jpg'
L1 = 2500
L2 = 2500
L3 = 2500
L4 = 900

print(get_angles_from_image(image_path, L1, L2, L3, L4))
