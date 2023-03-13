# Load the pre-trained model and tokenizer
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, Trainer
# Attach a new linear layer to the pre-trained model
import torch.nn as nn
# Prepare your data and train the model
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import os
from sklearn.metrics import classification_report
import numpy as np
from evaluation import argmax_prediction, metrics_report, metrics_report_for_emo_binary, FocalLoss
import datetime

emotion_label_policy = {'angry': 0, 'anger': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3, 'happines': 3, 'happiness': 3, 'excited': 3,
    'sad': 4, 'sadness': 4, 'frustrated': 4,
    'surprise': 5, 'surprised': 5, 
    'neutral': 6}    

def get_labels(test_file):
    # from evaluation.py
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    emotion_labels = []
    for doc in data.values():
        for utt in doc[0]:
            emotion_labels.append(utt['emotion'])
    
    return emotion_labels

def test(original_model, model, testfile, log_label, start_time, epoch):
    # Load the test dataset
    test_dataset = get_dataset_from_file(testfile)
    label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    # Process labels to number tensor
    labels_text = get_labels(testfile)
    labels_number = [emotion_label_policy[i] for i in labels_text]      # 'text labels from data' to 'number labels'
    labels_tensor = torch.tensor(labels_number)
    
    dataset = get_dataset_from_file(testfile)
    dataloader = DataLoader(dataset, shuffle=False)
    
    with open(testfile) as f:
        data = json.load(f)
        
    texts = []
    labels_text = []
    for doc in data.values():
        for utt in doc[0]:
            texts.append(utt['utterance'])
            labels_text.append(utt['emotion'])
    encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    labels_number = [emotion_label_policy[i] for i in labels_text]      # 'text labels from data' to 'number labels'

    input_ids = encoded_texts['input_ids'].to(device)
    attention_masks = encoded_texts['attention_mask'].to(device)
    
    model.eval()
    model_outputs = original_model(input_ids, attention_masks)[0]
    predictions = model(model_outputs)
    
    report = metrics_report(predictions, labels_tensor, label_)
    report += '\n'+metrics_report_for_emo_binary(predictions, labels_tensor)+'\n'
    print(report)
    
    with open(f'log/test_{log_label}-{str(start_time)}.txt', 'a') as f:
        f.write(f'Epoch: {epoch} | Test Report About: {log_label}\n')
        f.write(report)
    
# def argmax_prediction(pred_y, true_y):
#     pred_argmax = torch.argmax(pred_y, dim=1).cpu()
#     true_y = true_y.cpu()
#     return pred_argmax, true_y

# def metrics_report(pred_y, true_y, label):
#     pred_y, true_y = argmax_prediction(pred_y, true_y)
#     available_label = sorted(list(set(true_y.tolist() + pred_y.tolist())))

#     class_name = list(label[available_label])
#     return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4)
    

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

def set_model_with_added_layer(model_name, original_topology=7, num_classes=7):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
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
    
    # gpu use setting 
    gpus = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(_) for _ in gpus])
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    device = 'cuda:0'
    
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

    # test_filename = 'data_fold/data_1/data_1_test.json'
    
    for train_filename, test_filename, data_label in zip(train_data_list, test_data_list, data_label_list):
        # Experiment Arguments setting 
        num_epoch = 30
        original_topology = 7
        use_wandb = False
        wandb_project_name = f'j-hartmann-distilroberta-base-fine-tuned_on_{data_label}'
        # Initialize WandB
        if use_wandb:
            wandb.init(project=wandb_project_name)
        
        original_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        original_model.cuda()
        # Freeze the pre-trained model's parameters
        for param in original_model.parameters():
            param.requires_grad = False
        model = set_model_with_added_layer(model_name, original_topology=original_topology, num_classes=7)
        model = model.cuda()
        
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        dataset = get_dataset_from_file(train_filename)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
    
        i = 0
        start_time = datetime.datetime.now()
        for epoch in range(num_epoch):
            running_loss = 0.0
            for inputs, masks, labels in tqdm(dataloader, desc=f"train | Epoch {epoch+1}"):
                
                optimizer.zero_grad()
                
                model_outputs = original_model(inputs, masks)[0]
                outputs = model(model_outputs)
                
                criterion = FocalLoss(gamma=2)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                
                if use_wandb:
                    if i % 100 == 99:  # Log every 100 mini-batches
                        wandb.log({'train_loss': running_loss / 100})
                        running_loss = 0.0
                    i+=1
                
            epoch_loss = running_loss / len(dataset)
            
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epoch, epoch_loss))
            
            test(original_model, model, test_filename, data_label, start_time, epoch+1)
        # Finish the WandB run
        if use_wandb:
            wandb.finish()
            
        torch.save(model.state_dict(), f'model/model_j-hartmann-base_fine-tuned_{data_label}.pt')
        del model

