from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, AutoModelForSequenceClassification

class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}

def get_dataset(model_name, test_file):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    text_list = ['I like that', 'That is annoying', 'This is great!', 'Wouldn´t recommend it.']
    tokenized_texts = tokenizer(text_list,truncation=True,padding=True)
    dataset = SimpleDataset(tokenized_texts)
    return dataset