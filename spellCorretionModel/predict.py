import torch
import joblib
from models import SpellCorrectionModel
from config import Config

def load_model():
    vectorizer = joblib.load(Config.vectorizer_path)
    label_encoder = joblib.load(Config.label_encoder_path)
    model = SpellCorrectionModel(Config.input_size, Config.hidden_size, Config.output_size)
    model.load_state_dict(torch.load(Config.model_path))
    model.eval()
    return model, vectorizer, label_encoder

def correct_spelling(word, model, vectorizer, label_encoder):
    word_vec = vectorizer.transform([word]).toarray()
    word_tensor = torch.tensor(word_vec, dtype=torch.float32)
    with torch.no_grad():
        output = model(word_tensor)
        _, predicted = torch.max(output, 1)
        corrected_word = label_encoder.inverse_transform([predicted.item()])[0]
    return corrected_word

def correct_spelling_batch(words, model, vectorizer, label_encoder):
    word_vecs = vectorizer.transform(words).toarray()
    word_tensors = torch.tensor(word_vecs, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(word_tensors)
        _, predicted = torch.max(outputs, 1)
        corrected_words = label_encoder.inverse_transform(predicted.numpy())
    return corrected_words

if __name__ == "__main__":
    model, vectorizer, label_encoder = load_model()
    print("Система исправления орфографических ошибок. Введите 'exit' для выхода.")
    while True:
        word = input("Введите слово с ошибкой: ").strip()
        if word.lower() == 'exit':
            break

        corrected_word = correct_spelling(word, model, vectorizer, label_encoder)
        print(f"Исправленное слово: {corrected_word}")
