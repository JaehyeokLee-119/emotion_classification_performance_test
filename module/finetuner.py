from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, Trainer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import os
import numpy as np
from evaluation import metrics_report, metrics_report_for_emo_binary, FocalLoss
from dataset import get_bulk_texts_and_labels
import datetime

emotion_label_policy = {'angry': 0, 'anger': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3, 'happines': 3, 'happiness': 3, 'excited': 3,
    'sad': 4, 'sadness': 4, 'frustrated': 4,
    'surprise': 5, 'surprised': 5, 
    'neutral': 6}    

def get_dataset_from_file(filename):
    # Load the JSON dataset
    with open(filename) as f:
        data = json.load(f)
    texts = []
    labels_text = []
    for doc in data.values():
        for utt in doc[0]:
            texts.append(utt['utterance'])
            labels_text.append(utt['emotion'])
    encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    # Process labels to number tensor
    labels_number = [emotion_label_policy[i] for i in labels_text]      # 'text labels from data' to 'number labels'

    input_ids = encoded_texts['input_ids'].to(device)
    attention_masks = encoded_texts['attention_mask'].to(device)
    labels = torch.tensor(labels_number).to(device)
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset

def test(original_model, model, testfile, log_label, start_time, epoch_num, type_label='test'):
    batch_size = 5
    
    label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    
    texts, labels_text = get_bulk_texts_and_labels(testfile) # 파일에서 utterance list와 emotion label list를 불러옴
    labels_tensor = torch.tensor([emotion_label_policy[i] for i in labels_text])      # Emotion text label을 각 감정에 해당하는 숫자 tensor로 바꾼다
    
    encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt') # tokenizer로 발화 문장을 encoding

    input_ids = encoded_texts['input_ids'].to(device)
    attention_masks = encoded_texts['attention_mask'].to(device)
    
    dataset = get_dataset_from_file(testfile)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        predictions, labels_tensor = [list() for _ in range(2)]
        loss_overall = 0.0 # logging에 쓸 loss 값 저장용
        for inputs, masks, labels in tqdm(dataloader, desc=f"Test for {epoch_num+1}"):
            optimizer.zero_grad()
            
            model_outputs = original_model(inputs, masks)[0]
            outputs = model(model_outputs)
            
            criterion = FocalLoss(gamma=2)
            loss = criterion(outputs, labels)
            
            loss_overall += loss.item() * inputs.size(0)
            
            for pred in outputs:
                predictions.append(pred)
            for label in labels:
                labels_tensor.append(label)
    
    predictions = torch.stack(predictions).cpu()            # 펴서 tensor로 만들어줌
    labels_tensor = torch.stack(labels_tensor).cpu()        # 펴서 tensor로 만들어줌
    
    report = metrics_report(predictions, labels_tensor, label_)
    report += '\n'+metrics_report_for_emo_binary(predictions, labels_tensor)+'\n'
    print(report)
    
    # report를 파일에 저장
    with open(f'log/{type_label}_{log_label}-{str(start_time)}.txt', 'a') as f:
        f.write(f'Epoch: {epoch_num} | Test Report About: {log_label}\n')
        f.write(report)

def set_model_with_added_layer(model_name, original_topology=7, num_classes=7):
    '''
    Pre-trained Classificaiton 모델 (LM + classification layer) 위에다 또 linear layer를 얹어서
    추가된 linear layer에 분류를 학습
    이 함수는 추가 linear layer(added_model)를 리턴한다
    '''
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Freeze the pre-trained model's parameters
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(
        nn.Linear(in_features=original_topology, out_features=num_classes),
        # nn.ReLU(),
        # nn.Dropout(p=0.1),
        # nn.Linear(in_features=num_classes, out_features=num_classes)
    )
    added_model = model.classifier
    return added_model


def set_model_with_replaced_layer(model_name, num_classes=7):
    ''' 
    Pre-trained Classification 모델에서 LM만 가져와서 분류 역할은 새로운 linear layer만 맡게 된다
    그러므로 set_model_with_added_layer에 비해 분류 layer가 하나 적다
    '''
    model = AutoModel.from_pretrained(model_name)
    
    # Freeze the pre-trained model's parameters
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(
        nn.Linear(in_features=original_topology, out_features=num_classes),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(in_features=num_classes, out_features=num_classes)
    )
    added_model = model.classifier
    return added_model


if __name__ == '__main__':
    # setting 
    gpus = "1"
    device = 'cuda:0'
    # Experiment Arguments setting 
    num_epoch = 30
    use_wandb = False
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(_) for _ in gpus])
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    
    # Load model from huggingface
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_data_list = [
        'data_fold/data_0/dailydialog_train.json',
        * [f'data_fold/data_{fold_}/data_{fold_}_train.json' for fold_ in range(1, 5)]
    ]
    test_data_list = [
        'data_fold/data_0/dailydialog_test.json',
        * [f'data_fold/data_{fold_}/data_{fold_}_test.json' for fold_ in range(1, 5)]
    ]
    data_label_list = ['-original_data_DailyDialog', *[f'-data_{fold_}_DailyDialog' for fold_ in range(1, 5)]]

    for train_filename, test_filename, data_label in zip(train_data_list, test_data_list, data_label_list):
        # Initialize WandB
        if use_wandb:
            wandb_project_name = f'j-hartmann-distilroberta-base-fine-tuned_on_{data_label}'
            wandb.init(project=wandb_project_name)
        
        original_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        original_model.cuda()
        original_topology = len(original_model.config.label2id) # original model's output size
        
        # Freeze the pre-trained model's parameters
        for param in original_model.parameters():
            param.requires_grad = False
        model = set_model_with_added_layer(model_name, original_topology=original_topology, num_classes=7)
        model = model.cuda()
        
        optimizer = optim.Adam(model.parameters(), lr=5e-5)
        dataset = get_dataset_from_file(train_filename)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
        
        i = 0
        start_time = datetime.datetime.now()
        
        for epoch in range(num_epoch):
            loss_overall = 0.0 # logging에 쓸 loss 값 저장용
            loss_wandb = 0.0
            '''
            학습하면서 train 데이터에 대해 logging도 수행
            '''
            predictions, labels_tensor = [list() for _ in range(2)]
        
            for inputs, masks, labels in tqdm(dataloader, desc=f"train | Epoch {epoch+1}"):
                
                optimizer.zero_grad()
                model_outputs = original_model(inputs, masks)[0]
                outputs = model(model_outputs)
                
                criterion = FocalLoss(gamma=2)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                for pred in outputs:
                    predictions.append(pred)
                for label in labels:
                    labels_tensor.append(label)
                
                loss_overall += loss.item() * inputs.size(0)
                if use_wandb:
                    loss_wandb += loss.item() * inputs.size(0)
                    if i % 100 == 99:  # Log every 100 mini-batches
                        wandb.log({'train_loss': loss_wandb / 100})
                        loss_wandb = 0.0
                    i+=1
                    
            predictions = torch.stack(predictions).cpu()
            labels_tensor = torch.stack(labels_tensor).cpu()
             
            epoch_loss = loss_overall / len(dataset)
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epoch, epoch_loss))
            
            label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
            report = metrics_report(predictions, labels_tensor, label_)
            report += '\n'+metrics_report_for_emo_binary(predictions, labels_tensor)+'\n'
            # report를 파일에 저장
            with open(f'log/train_{data_label}-{str(start_time)}.txt', 'a') as f:
                f.write(f'Epoch: {epoch+1} | Test Report About: {data_label}\n')
                f.write(report)
            print(report)
            
            test(original_model, model, test_filename, data_label, start_time, epoch, type_label='test')
            
        # Finish the WandB run
        if use_wandb:
            wandb.finish()
            
        torch.save(model.state_dict(), f'model/model_j-hartmann-base_fine-tuned_{data_label}.pt')
        del model

