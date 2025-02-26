import os
import sys
import torch
from transformers import BertForMaskedLM, AdamW,  BertTokenizer
from utils import load_data, create_dataloader, save_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATHS, TRAINING_CONFIG, DATA_PATHS

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    train_data = load_data(DATA_PATHS['splits']['train'])
    model = BertForMaskedLM.from_pretrained(MODEL_PATHS['pretrained'])
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATHS['pretrained'])
    model.to(device)
    train_dataloader = create_dataloader(DATA_PATHS['splits']['train'], tokenizer, batch_size=TRAINING_CONFIG['batch_size'])
    optimizer = AdamW(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
    for epoch in range(TRAINING_CONFIG['epochs']):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')
    save_model(model, tokenizer, MODEL_PATHS['trained'])

if __name__ == '__main__':
    train_model()
