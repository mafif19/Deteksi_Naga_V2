import torch
from torchvision import models

def load_model(model_path: str):
    model = models.resnet50(pretrained=False)
    num_classes = 5
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
