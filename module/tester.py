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
    '''
    Tester는 감정 분류 체계가 DailyDialog와 같은 모델을 테스트할 때 사용
    - init(): 모델 이름, 테스트 파일들, 모델 라벨, 데이터 라벨을 받고, set_model()을 실행한다
        - set_model(): self.model_name에 해당하는 모델을 불러온다
    - report_jhartmann(): j-hartmann류의 모델을 테스트해서 Report를 생성한다
    - save_report(): Report를 파일로 저장한다 (폴더, 이름)
    - run(): 위 과정들을 포함해서 테스트를 실행한다
        각 테스트용 파일에 대해 모델을 테스트하고 report를 생성해서 저장한다
    '''
    
    
    def __init__(self, model_name, model_label, test_files, data_labels):
        self.model_name = model_name
        self.set_model()
        self.testfiles = test_files
        self.model_label = model_label
        self.data_labels = data_labels
        
    def set_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)  
    
    def report_jhartmann(self, predictions, testfile):
        # j-hartmann labels
        # Dataset 기준 label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
        emotion_label = [0, 1, 2, 3, 5, 6, 4] # ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
        emotion_label_policy = {'angry': emotion_label[0], 'anger': emotion_label[0],
            'disgust': emotion_label[1],
            'fear': emotion_label[2],
            'happy': emotion_label[3], 'happines': emotion_label[3], 'happiness': emotion_label[3], 'excited': emotion_label[3],
            'sad': emotion_label[4], 'sadness': emotion_label[4], 'frustrated': emotion_label[4],
            'surprise': emotion_label[5], ' surprised': emotion_label[5], 
            'neutral': emotion_label[6]}
        
        # Process labels to number tensor
        labels_text = get_labels(testfile)
        labels_number = [emotion_label_policy[i] for i in labels_text]      # 'text labels from data' to 'number labels'
        labels_tensor = torch.tensor(labels_number)
        
        report = metrics_report(predictions, labels_tensor, label_)
        report += '\n' + metrics_report_for_emo_binary(predictions, labels_tensor, neutral_num=4)
        return report
    
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
        '''
        Tester의 역할: Classification model을 테스트
        - Classification model: input으로 text를 받아서, 감정을 분류 점수를 리턴 (7개의 점수)         
        '''
        for tf, dl in zip(self.testfiles, self.data_labels):
            # Dataset
            prediction_dataset = get_dataset(model_name=self.model_name, datafile=tf)
            predictions = torch.tensor(trainer.predict(prediction_dataset)[0])
        
            # Calculate report 
            if (self.model_name == "j-hartmann/emotion-english-distilroberta-base"):
                report = self.report_jhartmann(predictions, tf)
            elif (self.model_name == "j-hartmann/emotion-english-roberta-large"):
                report = self.report_jhartmann(predictions, tf)
            else:
                report = 'None'
        
            # Print report
            self.save_report(report, dir_name='log', data_label=dl)
        