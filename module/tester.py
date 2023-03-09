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
        
        # Dataset
        pred_dataset = get_dataset(model_name=self.model_name, test_file=self.testfile)
        
        # Get Emotion true (str) from testfile
        labels = get_labels(self.testfile)
        
            # j-hartmann/emotion-english-distilroberta-base labels
        emotion_label = [0, 1, 2, 3, 4, 6, 5]
        label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise'])
        emotion_label_policy = {'angry': emotion_label[0], 'anger': emotion_label[0],
            'disgust': emotion_label[1],
            'fear': emotion_label[2],
            'happy': emotion_label[3], 'happines': emotion_label[3], 'happiness': emotion_label[3], 'excited': emotion_label[3],
            'sad': emotion_label[4], 'sadness': emotion_label[4], 'frustrated': emotion_label[4],
            'surprise': emotion_label[5], ' surprised': emotion_label[5], 
            'neutral': emotion_label[6]}
        
        # Process labels to number tensor
        for i in range(len(labels)):
            labels[i] = emotion_label_policy[labels[i]]
        labels_tensor = torch.tensor(labels)
        predictions = torch.tensor(trainer.predict(pred_dataset)[0])
        report = metrics_report(predictions, labels_tensor, label_)
        
        
        pass
        