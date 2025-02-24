import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from collections import defaultdict
import argparse
from models import SpellCorrectionModel
from config import Config

# Функция для выбора устройства
def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    elif use_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(description="Обучение модели исправления орфографических ошибок.")
parser.add_argument("--gpu", action="store_true", help="Использовать GPU для обучения (если доступно).")
args = parser.parse_args()

# Выбор устройства
device = get_device(args.gpu)
print(f"Используемое устройство: {device}")

# Загрузка данных
data = pd.read_csv('../datasets/orfo.csv', delimiter=';')
X = data['MISTAKE']
y = data['CORRECT']
weights = data['WEIGHT']

# Векторизация текста
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
X_vec = vectorizer.fit_transform(X)

# Кодирование меток
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Вычисление весов классов
class_weights = defaultdict(float)
for correct_word, weight in zip(y, weights):
    class_weights[correct_word] += weight

class_weights_tensor = torch.tensor(
    [class_weights[word] for word in label_encoder.classes_],
    dtype=torch.float32
).to(device)
class_weights_tensor = 1.0 / class_weights_tensor  # Инвертируем веса
class_weights_tensor = class_weights_tensor / class_weights_tensor.sum()  # Нормализуем

# Обновление конфигурации
Config.input_size = X_vec.shape[1]
Config.output_size = len(label_encoder.classes_)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_vec, y_encoded, test_size=0.2, random_state=42)

# Преобразование в тензоры PyTorch
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Создание DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

# Создание модели и перенос на устройство
model = SpellCorrectionModel(Config.input_size, Config.hidden_size, Config.output_size).to(device)

# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

# Обучение модели
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

    print(f"Эпоха [{epoch + 1}/{Config.num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Сохранение модели и вспомогательных объектов
torch.save(model.state_dict(), Config.model_path)
joblib.dump(vectorizer, Config.vectorizer_path)
joblib.dump(label_encoder, Config.label_encoder_path)

print("Обучение завершено и модель сохранена!")
