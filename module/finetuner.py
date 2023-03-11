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

from module.tester import report_general


def test():
    pass

def get_dataset_from_file(filename):
    emotion_label_policy = {'angry': 0, 'anger': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3, 'happines': 3, 'happiness': 3, 'excited': 3,
        'sad': 4, 'sadness': 4, 'frustrated': 4,
        'surprise': 5, 'surprised': 5, 
        'neutral': 6}    
    
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

def set_model_with_added_layer(model_name, num_classes=7):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Freeze the pre-trained model's parameters
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(
        nn.Linear(in_features=model.config.hidden_size, out_features=num_classes),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(in_features=num_classes, out_features=num_classes)
    )
    
    
if __name__ == '__main__':
    # gpu use setting 
    gpus = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(_) for _ in gpus])
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    device = 'cuda:0'
    
    # Load model from huggingface
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    model = set_model_with_added_layer(model_name, num_classes=7)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Experiment Arguments setting 
    num_epoch = 20
    use_wandb = False
    wandb_project_name = 'my-project'
    # Initialize WandB
    if use_wandb:
        wandb.init(project=wandb_project_name)
    
    # Extract the text data from the data file
    # Load the JSON dataset
    train_filename = 'data_fold/data_1/data_1_train.json'
    test_filename = 'data_fold/data_1/data_1_test.json'
    
    model = model.cuda()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    dataset = get_dataset_from_file(train_filename)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
    
    i = 0
    for epoch in range(num_epoch):
        running_loss = 0.0
        for inputs, masks, labels in tqdm(dataloader, desc=f"train | Epoch {epoch+1}"):
            
            optimizer.zero_grad()
            
            one_hot_labels = F.one_hot(labels, num_classes=7)
            outputs = model(inputs, masks)[0]
            breakpoint()
            loss = criterion(outputs, one_hot_labels)
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
        
    # Finish the WandB run
    if use_wandb:
        wandb.finish()
    torch.save(model.state_dict(), 'model/model_1.pt')


