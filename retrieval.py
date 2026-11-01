import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import hashlib
import numpy as np
import faiss
from PIL import Image
from sentence_transformers import SentenceTransformer

from embedding_model import load_encoder
from config import (
    INDEX_DIR, CHECKPOINT_PATH, TEXT_MODEL, MIN_IMAGE_SCORE, MIN_TEXT_SCORE,
    MAX_RESULTS, UNCERTAIN_THRESHOLD
)


def compute_md5(filepath: str) -> str:
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class VisionRetriever:
    def __init__(self, index_dir: str = None):
        if index_dir is None:
            index_dir = INDEX_DIR
        
        self.index_dir = index_dir
        
        config_path = os.path.join(index_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config not found at {config_path}. Run build_index.py first."
            )
        
        with open(config_path) as f:
            self.config = json.load(f)
        with open(os.path.join(index_dir, "metadata.json")) as f:
            self.metadata = json.load(f)
        
        self.image_index = faiss.read_index(os.path.join(index_dir, "image_index.faiss"))
        self.text_index = faiss.read_index(os.path.join(index_dir, "text_index.faiss"))
        
        checkpoint_path = self.config.get("checkpoint_path", CHECKPOINT_PATH)
        self.encoder = load_encoder(checkpoint_path)
        self.text_model = SentenceTransformer(TEXT_MODEL)
        self.class_names = self.encoder.class_names

    def is_stale(self) -> bool:
        stored_md5 = self.config.get("checkpoint_md5", "")
        if not stored_md5:
            return True
        current_md5 = compute_md5(self.config.get("checkpoint_path", CHECKPOINT_PATH))
        return current_md5 != stored_md5

    def query_by_image(self, path: str, k: int = 5, min_score: float = None) -> list:
        if not os.path.exists(path):
            return []
        
        img = Image.open(path).convert("RGB")
        vec = self.encoder.get_embedding(img)
        vec = vec.astype("float32").reshape(1, -1)
        faiss.normalize_L2(vec)
        
        search_k = min(max(k * 2, 20), self.image_index.ntotal)
        scores, indices = self.image_index.search(vec, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            s = float(score)
            item["image_score"] = s
            item["text_score"] = 0.0
            item["score"] = s
            results.append(item)
            
        top_result = results[0] if results else None
        is_uncertain = (top_result is not None and top_result.get("confidence", 1.0) < UNCERTAIN_THRESHOLD)
        for r in results:
            r["uncertain"] = is_uncertain
            
        if min_score is not None:
            results = [r for r in results if r["score"] >= min_score]
                
        return results[:k]

    def query_by_text(self, text: str, k: int = 5, min_score: float = None) -> list:
        if not text or not text.strip():
            return []
        
        text_clean = text.strip()
        text_lower = text_clean.lower()
        
        vec = self.text_model.encode([text_clean], convert_to_numpy=True)[0]
        vec = vec.astype("float32").reshape(1, -1)
        faiss.normalize_L2(vec)
        
        search_k = min(max(k * 3, 30), self.text_index.ntotal)
        scores, indices = self.text_index.search(vec, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            s = float(score)
            item["image_score"] = 0.0
            item["text_score"] = s
            item["score"] = s
            
            cap = item.get("caption", "").lower()
            fname = item.get("filename", "").lower()
            text_repr = item.get("text_repr", "").lower()
            
            # Prioritize direct matches in label, filename, or text representation
            direct_match = (
                text_lower == cap
                or text_lower in cap
                or cap in text_lower
                or text_lower in fname
                or text_lower in text_repr
            )
            item["direct_match"] = direct_match
            results.append(item)
            
        # Sort direct matches first, then by semantic similarity score
        results.sort(
            key=lambda x: (
                1 if (x["direct_match"] or x["score"] > 0.85) else 0,
                x["score"]
            ),
            reverse=True
        )
        
        if min_score is not None:
            results = [r for r in results if r["score"] >= min_score]
                
        return results[:k]

    def query_hybrid(self, path: str, text: str = "", k: int = 5, alpha: float = 0.7, min_score: float = None) -> list:
        has_img = path and os.path.exists(path)
        has_txt = bool(text and text.strip())
        
        if not has_img and not has_txt:
            return []
        if not has_img:
            return self.query_by_text(text, k=k, min_score=min_score)
        if not has_txt:
            return self.query_by_image(path, k=k, min_score=min_score)
            
        # Both image and text are present
        img = Image.open(path).convert("RGB")
        image_vec = self.encoder.get_embedding(img).astype("float32").reshape(1, -1)
        faiss.normalize_L2(image_vec)
        
        text_clean = text.strip()
        text_lower = text_clean.lower()
        text_vec = self.text_model.encode([text_clean], convert_to_numpy=True)[0].astype("float32").reshape(1, -1)
        faiss.normalize_L2(text_vec)
        
        n_total = len(self.metadata)
        img_scores, img_indices = self.image_index.search(image_vec, n_total)
        txt_scores, txt_indices = self.text_index.search(text_vec, n_total)
        
        img_score_map = {idx: s for s, idx in zip(img_scores[0], img_indices[0]) if idx != -1}
        txt_score_map = {idx: s for s, idx in zip(txt_scores[0], txt_indices[0]) if idx != -1}
        
        fused_results = []
        for idx in range(n_total):
            s_img = float(img_score_map.get(idx, 0.0))
            s_txt = float(txt_score_map.get(idx, 0.0))
            fused = alpha * s_img + (1.0 - alpha) * s_txt
            
            item = dict(self.metadata[idx])
            item["image_score"] = s_img
            item["text_score"] = s_txt
            item["score"] = fused
            
            cap = item.get("caption", "").lower()
            fname = item.get("filename", "").lower()
            text_repr = item.get("text_repr", "").lower()
            item["direct_match"] = (
                text_lower in cap
                or cap in text_lower
                or text_lower in fname
                or text_lower in text_repr
            )
            fused_results.append(item)
            
        fused_results.sort(
            key=lambda x: (
                1 if (x["direct_match"] and x["text_score"] > 0.75) else 0,
                x["score"]
            ),
            reverse=True
        )
        
        if min_score is not None:
            fused_results = [r for r in fused_results if r["score"] >= min_score]
                
        return fused_results[:k]

    def query_all_related(self, path: str = "", text: str = "", threshold: float = 0.5, 
                          max_results: int = 100, alpha: float = 0.7) -> list:
        results = self.query_hybrid(path, text, k=max_results, alpha=alpha)
        filtered = [r for r in results if r["score"] >= threshold]
        return filtered if filtered else results[:10]
