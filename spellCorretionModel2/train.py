from models import SpellCorrectionModel
from config import Config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
data = pd.read_csv('../datasets/error_dataset.csv')
X = data['Erroneous']
y = data['Correct']

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=5000)
X_vec = vectorizer.fit_transform(X)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

Config.input_size = X_vec.shape[1]
Config.output_size = len(label_encoder.classes_)
logging.info(f"input_size: {Config.input_size}")
logging.info(f"output_size: {Config.output_size}")

config = {
    "input_size": X_vec.shape[1],
    "output_size": len(label_encoder.classes_),
    "hidden_size": Config.hidden_size,
    "model_path": Config.model_path,
    "vectorizer_path": Config.vectorizer_path,
    "label_encoder_path": Config.label_encoder_path
}
joblib.dump(config, "config.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Используемое устройство: {device}")
X_train, X_test, y_train, y_test = train_test_split(X_vec, y_encoded, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
model = SpellCorrectionModel(Config.input_size, Config.hidden_size, Config.output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

logging.info("Начало обучения")
for epoch in range(Config.num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    logging.info(f"Эпоха [{epoch + 1}/{Config.num_epochs}], Loss: {epoch_loss:.4f}")

# Сохранение модели и вспомогательных объектов
torch.save(model.state_dict(), Config.model_path)
joblib.dump(vectorizer, Config.vectorizer_path)
joblib.dump(label_encoder, Config.label_encoder_path)
logging.info("Обучение завершено и модель сохранена!")
