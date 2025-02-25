import random
import csv
import re

def is_punctuation(token):
    return bool(re.match(r'^[^\w\s]+$', token))

def introduce_errors(word, num_errors=1):
    error_variants = []
    for _ in range(num_errors):
        if len(word) <= 1:
            error_variants.append(word)
            continue

        error_type = random.choice(['replace', 'delete', 'insert', 'swap'])
        if error_type == 'replace':
            pos = random.randint(0, len(word) - 1)
            new_char = random.choice('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
            erroneous_word = word[:pos] + new_char + word[pos+1:]

        elif error_type == 'delete':
            pos = random.randint(0, len(word) - 1)
            erroneous_word = word[:pos] + word[pos+1:]

        elif error_type == 'insert':
            pos = random.randint(0, len(word))
            new_char = random.choice('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
            erroneous_word = word[:pos] + new_char + word[pos:]

        elif error_type == 'swap':
            if len(word) >= 2:
                pos = random.randint(0, len(word) - 2)
                erroneous_word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
            else:
                erroneous_word = word
        error_variants.append(erroneous_word)

    return error_variants

def create_error_dataset(conllu_file, output_file, num_errors_per_word=3):
    data = []
    with open(conllu_file, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('#') or line.strip() == '':
                continue

            columns = line.strip().split('\t')
            if len(columns) >= 3:
                token = columns[1]
                lemma = columns[2]
                token = token.lower()
                lemma = lemma.lower()
                if is_punctuation(token):
                    continue

                error_variants = introduce_errors(token, num_errors_per_word)
                for erroneous_token in error_variants:
                    data.append((erroneous_token, token))

    with open(output_file, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Erroneous', 'Correct'])
        writer.writerows(data)

conllu_file = '../datasets/ru_syntagrus-ud-train-a.conllu'
output_csv = 'error_dataset.csv'
create_error_dataset(conllu_file, output_csv, num_errors_per_word=3)
print(f"Датасет с ошибками сохранен в {output_csv}")
