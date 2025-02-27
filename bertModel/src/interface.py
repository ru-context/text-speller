import torch, os, sys
from transformers import BertForMaskedLM, BertTokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATHS, MODEL_PATHS, TRAINING_CONFIG

def correct_text(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
    corrected_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
    return corrected_text

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    print("Загрузка модели и токенизатора...")
    model = BertForMaskedLM.from_pretrained(MODEL_PATHS['trained'])
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATHS['trained'])
    model.to(device)

    while True:
        text = input("Введите текст для исправления (или 'exit' для выхода): ")
        if text.lower() == 'exit':
            break
        corrected_text = correct_text(model, tokenizer, text, device)
        print(f"Исправленный текст: {corrected_text}")

if __name__ == '__main__':
    main()
