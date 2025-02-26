import torch
from transformers import BertForMaskedLM, BertTokenizer
from torch.optim import AdamW
from utils import load_data, create_dataloader, save_model
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATHS, MODEL_PATHS, TRAINING_CONFIG

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    model = BertForMaskedLM.from_pretrained(MODEL_PATHS['pretrained'])
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATHS['pretrained'])
    model.to(device)
    train_data = load_data(DATA_PATHS['splits']['train'])
    train_dataloader = create_dataloader(DATA_PATHS['splits']['train'], tokenizer, batch_size=TRAINING_CONFIG['batch_size'])
    print(f"DataLoader создан. Количество батчей: {len(train_dataloader)}")
    optimizer = AdamW(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])

    print("Начало обучения...")
    for epoch in range(TRAINING_CONFIG['epochs']):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{TRAINING_CONFIG['epochs']}", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({"Loss": loss.item()})
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Средний Loss: {avg_loss}")
    print("Сохранение модели и токенизатора...")
    save_model(model, tokenizer, MODEL_PATHS['trained'])
    print(f"Модель и токенизатор сохранены в {MODEL_PATHS['trained']}")

if __name__ == '__main__':
    train_model()
