import logging
import os
import datetime
import numpy as np
import json
from pprint import pprint

from module.evaluation import metrics_report, metrics_report_for_emo_binary
from module.dataset import get_dataset, get_labels

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class Tester:
    def __init__(self, model_name, model_label, test_files, data_labels):
        self.model_name = model_name
        self.set_model()
        self.testfiles = test_files
        self.model_label = model_label
        self.data_labels = data_labels
        
    def set_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)  
    
    def report_jhartmann(self, predictions, testfile):
        labels_text = get_labels(testfile)
        
        # j-hartmann/emotion-english-distilroberta-base labels
        emotion_label = [0, 1, 2, 3, 5, 6, 4] 
        # Dataset 기준 label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
        label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
        emotion_label_policy = {'angry': emotion_label[0], 'anger': emotion_label[0],
            'disgust': emotion_label[1],
            'fear': emotion_label[2],
            'happy': emotion_label[3], 'happines': emotion_label[3], 'happiness': emotion_label[3], 'excited': emotion_label[3],
            'sad': emotion_label[4], 'sadness': emotion_label[4], 'frustrated': emotion_label[4],
            'surprise': emotion_label[5], ' surprised': emotion_label[5], 
            'neutral': emotion_label[6]}
        
        # Process labels to number tensor
        labels_number = [emotion_label_policy[i] for i in labels_text]      # 'text labels from data' to 'number labels'
        labels_tensor = torch.tensor(labels_number)
        
        predictions_to_index = torch.argmax(predictions, dim=1)             # 7d prediction to 1d scalar label
        predictions_to_text = [label_[i] for i in predictions_to_index.tolist()]
        
        report = metrics_report(predictions, labels_tensor, label_)
        report += '\n' + metrics_report_for_emo_binary(predictions, labels_tensor, neutral_num=4)
        return report
    
        # Codes for verification
        with open(self.testfile, 'r') as f:
            data = json.load(f)
    
        utterances = []
        for doc in data.values():
            for utt in doc[0]:
                utterances.append(utt['utterance'])
                
        verification = []
        for (a,b,c) in zip(utterances, predictions_to_text, labels_text):
            verification.append({'utterance': a, 'pred': b, 'true': c})
    
    def save_report(self, report, dir_name, data_label):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        inner_dir_name = dir_name+'/'+self.model_label
        if not os.path.exists(inner_dir_name):
            os.makedirs(inner_dir_name)
        
        file_name = self.model_label+'-'+data_label+'.txt'
        
        with open(os.path.join(inner_dir_name, file_name), 'w') as f:
            f.write(report)
    
    def run(self):
        trainer = Trainer(model=self.model)
        
        for tf, dl in zip(self.testfiles, self.data_labels):
            # Dataset
            print('testfile: ', tf, 'data_label:', dl)
            pred_dataset = get_dataset(model_name=self.model_name, test_file=tf)
            predictions = torch.tensor(trainer.predict(pred_dataset)[0])
        
            # Calculate report 
            if (self.model_name == "j-hartmann/emotion-english-distilroberta-base"):
                report = self.report_jhartmann(predictions, tf)
            elif (self.model_name == "j-hartmann/emotion-english-roberta-large"):
                report = self.report_jhartmann(predictions, tf)
            else:
                report = 'None'
        
            # Print report
            self.save_report(report, 'log', dl)
        