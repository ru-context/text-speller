import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class ErrorCorrectionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        error_text = self.data.iloc[idx]['error']
        correction_text = self.data.iloc[idx]['correction']

        # Токенизация текста с ошибкой и исправленного текста
        inputs = self.tokenizer(
            error_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        labels = self.tokenizer(
            correction_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).input_ids

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

def load_data(data_path):
    return pd.read_csv(data_path)

def create_dataloader(data_path, tokenizer, batch_size=16, max_length=128):
    dataset = ErrorCorrectionDataset(data_path, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def save_model(model, tokenizer, save_path):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Модель и токенизатор сохранены в {save_path}")
