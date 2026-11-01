import config

import os
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image

from retrieval import VisionRetriever, compute_md5
from config import CHECKPOINT_PATH, INDEX_DIR

IMAGE_SIZE = 64


def get_cifar100_dataset(root: str = "./data", train: bool = False):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
    return dataset


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    index_path = Path(INDEX_DIR)
    if not index_path.exists():
        print(f"ERROR: Index directory not found at {index_path}")
        print("Please run: python build_index.py --images_dir ./sample_images")
        return
    
    print("Loading CIFAR-100 test set...")
    try:
        test_dataset = get_cifar100_dataset(train=False)
        print(f"CIFAR-100 test set loaded: {len(test_dataset)} images")
    except Exception as e:
        print(f"Warning: Could not load CIFAR-100: {e}")
        return
    
    print("Loading retriever...")
    retriever = VisionRetriever(str(index_path))
    
    checkpoint_md5 = compute_md5(CHECKPOINT_PATH)
    stored_md5 = retriever.config.get("checkpoint_md5", "")
    is_stale = checkpoint_md5 != stored_md5
    
    print(f"\nChecking self-match rate...")
    self_match_scores = []
    sample_size = min(100, len(test_dataset))
    query_indices = random.sample(range(len(test_dataset)), sample_size)
    
    correct_self_match = 0
    for q_idx in query_indices:
        img_tensor = test_dataset[q_idx][0].numpy()
        img_np = img_tensor.transpose(1, 2, 0) * 0.5 + 0.5
        img_np = (img_np * 255).astype(np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            Image.fromarray(img_np).save(temp_path)
        
        try:
            results = retriever.query_by_image(temp_path, k=20)
            if results:
                top_score = results[0].get("score", 0)
                self_match_scores.append(top_score)
                if results[0].get("score", 0) > 0.98:
                    correct_self_match += 1
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    self_match_rate = correct_self_match / sample_size if sample_size > 0 else 0
    avg_self_match_score = np.mean(self_match_scores) if self_match_scores else 0
    
    print(f"Self-match rate: {self_match_rate*100:.1f}% (rank-1 score near 1.0)")
    if self_match_scores:
        print(f"Top-5 score stats: min={min(self_match_scores):.4f}, mean={np.mean(self_match_scores):.4f}, max={max(self_match_scores):.4f}")
    if self_match_rate < 0.95:
        print(f"WARNING: Self-match rate below 95% - index/model may be stale")
    
    prec5_img, prec10_img = [], []
    prec5_txt, prec10_txt = [], []
    prec5_hybrid, prec10_hybrid = [], []
    top5_scores_img, top5_scores_txt, top5_scores_hybrid = [], [], []
    top10_scores_img, top10_scores_txt, top10_scores_hybrid = [], [], []
    
    query_indices = random.sample(range(len(test_dataset)), min(100, len(test_dataset)))
    
    for q_idx in query_indices:
        true_label = test_dataset.classes[test_dataset.targets[q_idx]]
        
        img_tensor = test_dataset[q_idx][0].numpy()
        img_np = img_tensor.transpose(1, 2, 0) * 0.5 + 0.5
        img_np = (img_np * 255).astype(np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            Image.fromarray(img_np).save(temp_path)
        
        try:
            img_results = retriever.query_by_image(temp_path, k=20)
            
            if img_results:
                top5_scores_img.extend([r.get("score", 0) for r in img_results[:5]])
            
            captions_list = [r.get("caption", "") for r in img_results[:10]]
            correct_img = sum(1 for c in captions_list if c == true_label)
            prec5_img.append(correct_img / 5)
            prec10_img.append(correct_img / 10)
            
            txt_results = retriever.query_by_text(true_label, k=20)
            
            if txt_results:
                top5_scores_txt.extend([r.get("score", 0) for r in txt_results[:5]])
            
            captions_txt = [r.get("caption", "") for r in txt_results[:10]]
            correct_txt = sum(1 for c in captions_txt if c == true_label)
            prec5_txt.append(correct_txt / 5)
            prec10_txt.append(correct_txt / 10)
            
            hybrid_results = retriever.query_hybrid(temp_path, true_label, k=20, alpha=0.7)
            
            if hybrid_results:
                top5_scores_hybrid.extend([r.get("score", 0) for r in hybrid_results[:5]])
            
            captions_hybrid = [r.get("caption", "") for r in hybrid_results[:10]]
            correct_hybrid = sum(1 for c in captions_hybrid if c == true_label)
            prec5_hybrid.append(correct_hybrid / 5)
            prec10_hybrid.append(correct_hybrid / 10)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    report = {
        "is_stale": is_stale,
        "self_match_rate": float(self_match_rate),
        "avg_self_match_score": float(avg_self_match_score),
        "top5_min": float(min(self_match_scores)) if self_match_scores else 0,
        "top5_mean": float(np.mean(self_match_scores)) if self_match_scores else 0,
        "top5_max": float(max(self_match_scores)) if self_match_scores else 0,
        "precision_image_only": {
            "P@5": float(np.mean(prec5_img)) if prec5_img else 0,
            "P@10": float(np.mean(prec10_img)) if prec10_img else 0,
        },
        "precision_text_only": {
            "P@5": float(np.mean(prec5_txt)) if prec5_txt else 0,
            "P@10": float(np.mean(prec10_txt)) if prec10_txt else 0,
        },
        "precision_hybrid": {
            "P@5": float(np.mean(prec5_hybrid)) if prec5_hybrid else 0,
            "P@10": float(np.mean(prec10_hybrid)) if prec10_hybrid else 0,
        },
    }
    
    report_path = "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nEvaluation report saved to: {report_path}")
    print("\nSummary:")
    print(f"  Self-match rate: {self_match_rate*100:.1f}%")
    if self_match_scores:
        print(f"  Top-5 score: min={report['top5_min']:.4f}, mean={report['top5_mean']:.4f}, max={report['top5_max']:.4f}")
    print(f"  Precision@5 - Image only: {report['precision_image_only']['P@5']:.4f}")
    print(f"  Precision@5 - Text only: {report['precision_text_only']['P@5']:.4f}")
    print(f"  Precision@5 - Hybrid: {report['precision_hybrid']['P@5']:.4f}")
    
    if is_stale:
        print("\nWARNING: Index is stale! Rebuild with build_index.py")


if __name__ == "__main__":
    main()