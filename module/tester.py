import logging
import os
import datetime
import numpy as np
import json

from module.evaluation import log_metrics, metrics_report
from module.dataset import get_dataset, get_labels

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class Tester:
    def __init__(self, model_name, test_file):
        self.model_name = model_name
        self.set_model()
        self.testfile = test_file
        
    def set_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)  
        
    def run(self):
        trainer = Trainer(model=self.model)
        pred_dataset = get_dataset(model_name=self.model_name, test_file=self.testfile)
        labels = get_labels(self.testfile)
        predictions = trainer.predict(pred_dataset)[0]
        emotion_label = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise'])
        report = metrics_report(predictions, labels, emotion_label)
        pass
        