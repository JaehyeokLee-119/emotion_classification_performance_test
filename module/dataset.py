from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, AutoModelForSequenceClassification
import json

class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}

def get_dataset(model_name, test_file):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    utterances = []
    emotion_labels = []
    
    num_docs = 0
    num_utts = 0
    
    for doc in data.values():
        num_docs += 1
        for utt in doc[0]:
            utterances.append(utt['utterance'])
            emotion_labels.append(utt['emotion'])
            num_utts += 1
    
    tokenized_texts = tokenizer(utterances,truncation=True,padding=True)
    dataset = SimpleDataset(tokenized_texts)
    return dataset