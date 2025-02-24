class Config:
    # Гиперпараметры модели
    input_size = None
    hidden_size = 128
    output_size = None

    # Гиперпараметры обучения
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 32

    # Пути для сохранения и загрузки
    model_path = 'spell_correction_model.pth'
    vectorizer_path = 'vectorizer.pkl'
    label_encoder_path = 'label_encoder.pkl'
