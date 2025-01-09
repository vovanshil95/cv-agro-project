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


MAX_EPOCHS = 16
MIN_EPOCHS = 1
BATCH_SIZE = 16
IMAGE_SIZE = (128, 128)
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-5
devices = torch.cuda.device_count() if torch.cuda.is_available() else 1


class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.auc = torchmetrics.AUROC(task='multiclass', num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = F.cross_entropy(scores, y)
        return loss, scores, y


    def training_step(self, batch, batch_idx):        
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        auc_score = self.auc(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict(
            {'train_loss': loss, 'train_accuracy': accuracy, 'train_f1_score': f1_score, 'train_auc': auc_score},
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        
        return {'loss': loss, "scores": scores, "y": y}
    
    
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        auc_score = self.auc(scores, y)
        f1_score = self.f1_score(scores, y)

        to_log = {'val_loss': loss, 'val_accuracy': accuracy, 'val_f1_score': f1_score, 'val_auc': auc_score}
        logs.append(to_log)
        
        self.log_dict(to_log,
                      on_step=False, on_epoch=True, prog_bar=True)
        return loss
    

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        auc_score = self.auc(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy, 'test_f1_score': f1_score, 'test_auc': auc_score})
        return loss
    

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = self.argmax(scores, dim=1)
    

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=LEARNING_RATE)


transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

entire_dataset = datasets.ImageFolder(root='data', transform=transform)
train_ds, val_ds, test_ds = random_split(entire_dataset, [0.7, 0.1, 0.2])

train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=False)


logs = []


model = NN(input_size=128*128*3, num_classes=3)
trainer = pl.Trainer(accelerator=accelerator, devices=devices, min_epochs=MIN_EPOCHS, max_epochs=MAX_EPOCHS)
trainer.fit(model, train_loader, val_loader)


trainer.validate(model, val_loader)
trainer.test(model, test_loader)


logs = pd.DataFrame(logs)


val_loss = logs.val_loss.apply(lambda el: el.item()).tolist()
val_accuracy = logs.val_accuracy.apply(lambda el: el.item()).tolist()
val_f1_score = logs.val_f1_score.apply(lambda el: el.item()).tolist()
val_auc = logs.val_auc.apply(lambda el: el.item()).tolist()


plt.figure(figsize=(24, 10))

plt.subplot(1, 3, 1)
plt.plot(val_loss, label='Validation Loss', color='orange')
plt.title('Loss')
plt.xlabel('batch num')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('batch num')
plt.ylabel('Accuracy')
plt.legend()


plt.subplot(1, 3, 3)
plt.plot(val_f1_score, label='Validation F1 score')
plt.title('F1 score')
plt.xlabel('batch num')
plt.ylabel('F1 score')
plt.legend()


plt.tight_layout()
plt.show()
