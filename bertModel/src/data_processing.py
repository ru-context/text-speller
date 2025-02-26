import os
import sys
import random
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATHS

def load_conllu(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        for line in f:
            if line.startswith('#'):
                continue
            if line.strip() == '':
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                parts = line.strip().split('\t')
                sentence.append(parts)
        if sentence:
            sentences.append(sentence)
    return sentences

def create_errors(tokens):
    error_tokens = []
    for token in tokens:
        if random.random() < 0.1:  # 10% вероятность ошибки
            token = token[:len(token)//2] + token[len(token)//2+1:]
        error_tokens.append(token)
    return error_tokens

def create_error_correction_pairs(sentences):
    pairs = []
    for sentence in sentences:
        original_tokens = [token[1] for token in sentence]
        error_tokens = create_errors(original_tokens)
        pairs.append((' '.join(error_tokens), ' '.join(original_tokens)))
    return pairs

def save_pairs(pairs, save_path):
    df = pd.DataFrame(pairs, columns=['error', 'correction'])
    df.to_csv(save_path, index=False)

def split_dataset(data_path, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    df = pd.read_csv(data_path)
    assert train_size + val_size + test_size == 1.0, "Сумма train_size, val_size и test_size должна быть равна 1.0"
    train_df, temp_df = train_test_split(df, train_size=train_size, random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=test_size/(val_size + test_size), random_state=random_state)
    train_df.to_csv(DATA_PATHS['splits']['train'], index=False)
    val_df.to_csv(DATA_PATHS['splits']['val'], index=False)
    test_df.to_csv(DATA_PATHS['splits']['test'], index=False)

    print(f"Данные успешно разделены:")
    print(f"- Обучающая выборка: {len(train_df)} строк")
    print(f"- Валидационная выборка: {len(val_df)} строк")
    print(f"- Тестовая выборка: {len(test_df)} строк")

if __name__ == '__main__':
    sentences = load_conllu(DATA_PATHS['raw'])
    pairs = create_error_correction_pairs(sentences)
    save_pairs(pairs, DATA_PATHS['processed'])
    split_dataset(DATA_PATHS['processed'])
