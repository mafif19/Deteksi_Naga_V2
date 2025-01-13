import torch

def predict(image_tensor, model, classes):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, class_idx = torch.max(probabilities, 0)
        return classes[class_idx], confidence.item()
