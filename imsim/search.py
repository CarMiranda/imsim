import faiss
import numpy as np
import pathlib
import torch
from PIL import Image
from imsim.model import get_model, extract_features
import tqdm


def search(features, index, n: int):
    embeddings = features.mean(dim=1)
    vector = embeddings.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)

    d, i = index.search(vector, n)

    return d[0], i[0]


def embed_and_search(index, model_name: str, images: list[pathlib.Path], top_n: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = get_model(model_name, device)

    results = []
    for image_path in tqdm.tqdm(images):
        image = Image.open(image_path).convert("RGB")
        features = extract_features(processor, model, image, device).last_hidden_state
        results += [search(features, index, top_n)]

    return results
