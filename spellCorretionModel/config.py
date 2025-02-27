class Config:
    # Гиперпараметры модели
    input_size = None
    hidden_size = 128
    output_size = None

    # Гиперпараметры обучения
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 32

    # Пути для сохранения и загрузки
    model_path = '../locmodels/spellCorrectionMdls/spell_correction_model.pth'
    vectorizer_path = '../locmodels/spellCorrectionMdls/vectorizer.pkl'
    label_encoder_path = '../locmodels/spellCorrectionMdls/label_encoder.pkl'
