import torch
from transformers import AutoImageProcessor, AutoModel


def get_model(name: str, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device)

    return processor, model


@torch.no_grad()
def extract_features(processor, model, image, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    return outputs
