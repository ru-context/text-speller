import torch
import joblib
from models import SpellCorrectionModel
config = joblib.load("config.pkl")

def load_model():
    vectorizer = joblib.load(config["vectorizer_path"])
    label_encoder = joblib.load(config["label_encoder_path"])
    model = SpellCorrectionModel(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        output_size=config["output_size"]
    )

    model.load_state_dict(torch.load(config["model_path"], map_location=torch.device('cpu')))
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

if __name__ == "__main__":
    model, vectorizer, label_encoder = load_model()
    print("Система исправления орфографических ошибок. Введите 'exit' для выхода.")
    while True:
        word = input("Введите слово с ошибкой: ").strip()
        if word.lower() == 'exit':
            break
        corrected_word = correct_spelling(word, model, vectorizer, label_encoder)
        print(f"Исправленное слово: {corrected_word}")
