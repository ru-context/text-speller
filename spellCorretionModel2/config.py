class Config:
    # Гиперпараметры модели
    input_size = None
    hidden_size = 128
    output_size = None

    # Гиперпараметры обучения
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 256

    # Пути для сохранения и загрузки
    model_path = '../locmodels/spellCorrectionMdls2/spell_correction_model.pth'
    vectorizer_path = '../locmodels/spellCorrectionMdls2/vectorizer.pkl'
    label_encoder_path = '../locmodels/spellCorrectionMdls2/label_encoder.pkl'
