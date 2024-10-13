import torch
import torch.nn as nn
from utils.config import ModelConfig, ControlConfig
# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)
config_control = ControlConfig()
config_model = ModelConfig()
model = SimpleModel()
torch.save(model.state_dict(), config_model.weights_yolo)
loaded_model = SimpleModel()
loaded_model.load_state_dict(torch.load(config_model.weights_yolo))
loaded_model.eval()  # Set to evaluation mode
