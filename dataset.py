import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class ABSADataset(Dataset):
    def __init__(self, texts, aspects, labels, tokenizer, max_length=128):
        self.texts = [f'{aspect[0] if aspect else "other"} [SEP] {text if text.strip() else "neutral"}' for aspect, text in zip(aspects, texts)]
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }