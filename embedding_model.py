"""
embedding_model.py
--------------------
Defines the CNN architecture used to convert an image into a vector
(embedding), and shared helper functions to load the trained model
and extract embeddings from any image. This is used by
train_embedding_model.py (to train it) and by build_index.py /
retrieval.py (to use it).

This model is trained FROM SCRATCH on CIFAR-100 by
train_embedding_model.py -- it is NOT a pretrained/downloaded
embedding model like CLIP.
"""

import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 32          # CIFAR-100 native size
EMBEDDING_DIM = 256      # size of the output vector stored in FAISS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Matches checkpoint training-time preprocessing
CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
CIFAR_STD = [0.2673, 0.2564, 0.2762]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ]
)


class ResidualBlock(nn.Module):
    """
    Standard residual block with optional stride for downsampling.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.skip(x))
        return out


class EmbeddingCNN(nn.Module):
    """
    ResNet-18 architecture for CIFAR-100:
    stem + 8 residual blocks (with stride=2 downsampling on blocks 2, 4, 6) ->
    global average pooling -> embedding layer (Linear+BN) -> classifier.
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, num_classes: int = 100):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

        self.blocks = nn.Sequential(
            ResidualBlock(64, 64, stride=1),    # block 0
            ResidualBlock(64, 64, stride=1),    # block 1
            ResidualBlock(64, 128, stride=2),   # block 2 (downsample to 16x16)
            ResidualBlock(128, 128, stride=1),  # block 3
            ResidualBlock(128, 256, stride=2),  # block 4 (downsample to 8x8)
            ResidualBlock(256, 256, stride=1),  # block 5
            ResidualBlock(256, 512, stride=2),  # block 6 (downsample to 4x4)
            ResidualBlock(512, 512, stride=1),  # block 7
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.embedding_layer = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embedding: bool = False):
        x = F.relu(self.stem(x))
        x = self.blocks(x)
        x = self.gap(x).flatten(1)
        embedding = self.embedding_layer(x)
        if return_embedding:
            return embedding
        logits = self.classifier(embedding)
        return logits, embedding


class ImageEncoder:
    """
    Loads the trained checkpoint once and exposes a simple
    get_embedding() / get_embedding_and_label() API for the rest
    of the project.
    """

    def __init__(self, checkpoint_path: str = "embedding_model.pth"):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Trained model not found at {checkpoint_path}. "
                "Run train_embedding_model.py first."
            )

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)

        self.embedding_dim = checkpoint["embedding_dim"]
        self.class_names = checkpoint["class_names"]
        self.val_accuracy = checkpoint.get("val_accuracy", 0.0)
        self.test_accuracy = checkpoint.get("test_accuracy", 0.0)
        self.arch = checkpoint.get("arch", "EmbeddingResNet")
        self.model_version = checkpoint.get("version", 1)
        self.checkpoint_mtime = checkpoint.get("checkpoint_mtime", 0)
        self.checkpoint_md5 = checkpoint.get("checkpoint_md5", "")

        self.model = EmbeddingCNN(
            embedding_dim=self.embedding_dim, num_classes=len(self.class_names)
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(DEVICE)
        self.model.eval()

    def _preprocess(self, image: Image.Image):
        return TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    @torch.no_grad()
    def get_embedding(self, image: Image.Image) -> np.ndarray:
        x = self._preprocess(image)
        embedding = self.model(x, return_embedding=True)
        return embedding.cpu().numpy()[0].astype("float32")

    @torch.no_grad()
    def get_embedding_and_label(self, image: Image.Image):
        """Returns (embedding_vector, predicted_class_name, confidence)."""
        x = self._preprocess(image)
        logits, embedding = self.model(x, return_embedding=False)
        probs = F.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
        label = self.class_names[pred_idx.item()]
        return (
            embedding.cpu().numpy()[0].astype("float32"),
            label,
            float(conf.item()),
        )


def load_encoder(checkpoint_path: str = "embedding_model.pth") -> ImageEncoder:
    return ImageEncoder(checkpoint_path)
