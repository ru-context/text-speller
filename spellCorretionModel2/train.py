import torch
import joblib
from models import SpellCorrectionModel
from config import Config
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
def split_dataset(data, num_parts):
    part_size = len(data) // num_parts
    parts = []
    for i in range(num_parts):
        start = i * part_size
        end = (i + 1) * part_size if i < num_parts - 1 else len(data)
        parts.append(data[start:end])
    return parts

def train_on_part(part_data, model, vectorizer, label_encoder, device):
    X = part_data['Erroneous']
    y = part_data['Correct']

    X_vec = vectorizer.transform(X)
    y_encoded = label_encoder.transform(y)
    X_tensor = torch.tensor(X_vec.toarray(), dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)

    for epoch in range(Config.num_epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(loader, desc=f"Эпоха {epoch + 1}/{Config.num_epochs}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)
        logging.info(f"Эпоха [{epoch + 1}/{Config.num_epochs}], Loss: {epoch_loss:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Используемое устройство: {device}")
    if not os.path.exists("config.pkl"):
        logging.info("Файл config.pkl не найден. Создаем новый конфиг.")
        config = {
            "input_size": None,
            "output_size": None,
            "hidden_size": Config.hidden_size,
            "model_path": Config.model_path,
            "vectorizer_path": Config.vectorizer_path,
            "label_encoder_path": Config.label_encoder_path
        }
    else:
        logging.info("Загружаем существующий config.pkl.")
        config = joblib.load("config.pkl")

    data = pd.read_csv('../datasets/error_dataset.csv')
    num_parts = 5
    parts = split_dataset(data, num_parts)

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=5000)
    vectorizer.fit(data['Erroneous'])
    label_encoder = LabelEncoder()
    label_encoder.fit(data['Correct'])

    model = SpellCorrectionModel(
        input_size=vectorizer.transform(["dummy"]).shape[1],
        hidden_size=Config.hidden_size,
        output_size=len(label_encoder.classes_)
    ).to(device)

    config["input_size"] = vectorizer.transform(["dummy"]).shape[1]
    config["output_size"] = len(label_encoder.classes_)
    joblib.dump(config, "config.pkl")
    for part_num, part_data in enumerate(parts, 1):
        logging.info(f"Обучение на части {part_num}...")
        train_on_part(part_data, model, vectorizer, label_encoder, device)
        torch.save(model.state_dict(), config["model_path"])
        logging.info(f"Модель сохранена после части {part_num}.")

    joblib.dump(vectorizer, config["vectorizer_path"])
    joblib.dump(label_encoder, config["label_encoder_path"])
    logging.info("Обучение завершено и модель сохранена!")


if __name__ == "__main__":
    main()
