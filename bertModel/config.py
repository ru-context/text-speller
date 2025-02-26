DATA_PATHS = {
    'raw': 'data/raw/ru_syntagrus-ud-train-a.conllu',
    'processed': 'data/processed/error_correction_pairs.csv',
    'splits': {
        'train': 'data/splits/train.csv',
        'val': 'data/splits/val.csv',
        'test': 'data/splits/test.csv',
    }
}

MODEL_PATHS = {
    'pretrained': 'DeepPavlov/rubert-base-cased',
    'trained': 'models/trained/spelling_correction_model/'
}

TRAINING_CONFIG = {
    'batch_size': 128,
    'learning_rate': 5e-5,
    'epochs': 3,
    'max_length': 128
}
