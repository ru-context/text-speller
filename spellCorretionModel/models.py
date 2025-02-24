import torch.nn as nn
import torch.nn.functional as F

# Модель для поиска ошибок в тексте
class SpellCorrectionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SpellCorrectionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
