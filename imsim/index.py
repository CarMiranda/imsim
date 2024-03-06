import pathlib
from PIL import Image
import faiss
import numpy as np
import torch
from imsim.model import get_model, extract_features
import tqdm


def add_vector_to_index(embedding: torch.Tensor, index):
    vector = embedding.detach().cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vector)
    index.add(vector)


def update_index(index, model_name: str, images: list[pathlib.Path]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = get_model(model_name, device)

    for image_path in tqdm.tqdm(images):
        image = Image.open(image_path).convert("RGB")
        features = extract_features(processor, model, image, device).last_hidden_state
        add_vector_to_index(features.mean(dim=1), index)
