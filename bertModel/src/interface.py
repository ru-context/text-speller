import torch
from transformers import BertForMaskedLM, BertTokenizer
from config import MODEL_PATHS

def correct_text(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
    corrected_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
    return corrected_text

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    model = BertForMaskedLM.from_pretrained(MODEL_PATHS['trained'])
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATHS['trained'])

    model.to(device)
    text = "Привет, как дила?"
    corrected_text = correct_text(model, tokenizer, text, device)
    print(f"Исходный текст: {text}")
    print(f"Исправленный текст: {corrected_text}")
