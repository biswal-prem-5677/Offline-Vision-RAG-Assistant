import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import config
import json
import hashlib
import time
import re
from pathlib import Path

import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from embedding_model import load_encoder
from config import INDEX_DIR, CHECKPOINT_PATH, TEXT_MODEL

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_md5(filepath: str) -> str:
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def extract_clean_name(filename: str) -> str:
    base = os.path.splitext(filename)[0].lower()
    if base.startswith("screenshot") or base.startswith("whatsapp"):
        return ""
    cleaned = re.sub(r'[\d_-]+', ' ', base).strip()
    return cleaned


def collect_image_paths(images_dir: str):
    paths = []
    for root, _, files in os.walk(images_dir):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def build_index(images_dir: str, checkpoint_path: str = None):
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATH
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    out_dir = INDEX_DIR
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading image encoder...")
    encoder = load_encoder(checkpoint_path)
    
    image_paths = collect_image_paths(images_dir)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    
    image_vectors = []
    metadata = []
    captions = []
    
    for p in tqdm(image_paths, desc="Encoding images with CNN"):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        
        vec, label, confidence = encoder.get_embedding_and_label(img)
        filename = os.path.basename(p)
        clean_name = extract_clean_name(filename)
        
        text_repr = f"{label} {clean_name}".strip() if clean_name else label
        
        image_vectors.append(vec)
        captions.append(text_repr)
        metadata.append({
            "path": p,
            "filename": filename,
            "caption": label,
            "text_repr": text_repr,
            "confidence": round(confidence, 4),
        })
    
    image_vectors = np.vstack(image_vectors).astype("float32")
    faiss.normalize_L2(image_vectors)
    image_index = faiss.IndexFlatIP(image_vectors.shape[1])
    image_index.add(image_vectors)
    
    print(f"Loading text encoder ({TEXT_MODEL})...")
    text_model = SentenceTransformer(TEXT_MODEL)
    text_vectors = text_model.encode(captions, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(text_vectors)
    text_index = faiss.IndexFlatIP(text_vectors.shape[1])
    text_index.add(text_vectors)
    
    faiss.write_index(image_index, os.path.join(out_dir, "image_index.faiss"))
    faiss.write_index(text_index, os.path.join(out_dir, "text_index.faiss"))
    
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    config_info = {
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "text_model_name": TEXT_MODEL,
        "embedding_dim": int(image_vectors.shape[1]),
        "text_dim": int(text_vectors.shape[1]),
        "count": len(metadata),
        "checkpoint_mtime": os.path.getmtime(checkpoint_path),
        "checkpoint_md5": compute_md5(checkpoint_path),
        "build_time": time.time(),
    }
    
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_info, f, indent=2)
    
    print(f"Done. Indexed {len(metadata)} images (image vectors: {image_index.ntotal}, text vectors: {text_index.ntotal}).")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=CHECKPOINT_PATH)
    args = parser.parse_args()
    build_index(args.images_dir, args.checkpoint_path)