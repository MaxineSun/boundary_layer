from torch import nn
import torch.nn.functional as F


class predicted_model(nn.Module):  # 4 layers of MLP, not a good model actually
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3):
        super(predicted_model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out
