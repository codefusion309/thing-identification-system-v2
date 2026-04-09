"""
identifier.py - Embedding-Based Image Identification

Loads mean class embeddings from disk.
Identifies images via cosine similarity — no retraining ever needed.
Corrections update the class mean embedding incrementally in ~50ms.

Workflow:
  identify()       : image → embedding → cosine similarity vs all classes → top match
  add_correction() : new image → embedding → update running mean for that class
"""

import os
import json
import torch
from torchvision import transforms
from PIL import Image

from model import load_backbone, get_embedding


class Identifier:
    def __init__(self, embeddings_path: str, counts_path: str):
        """
        Args:
            embeddings_path: Path to saved_model/embeddings.pt
            counts_path:     Path to saved_model/counts.json
        """
        self.embeddings_path  = embeddings_path
        self.counts_path      = counts_path
        self.backbone         = None
        self.class_embeddings = {}   # { class_name: tensor(1, 1280) }
        self.class_counts     = {}   # { class_name: int }
        self.device           = torch.device("cpu")
        self.reload()

    def reload(self):
        """
        Load backbone and embeddings from disk.
        Called at startup and after a full index rebuild (POST /train).
        """
        self.backbone = load_backbone()
        self.backbone.to(self.device)
        self.backbone.eval()

        if os.path.exists(self.embeddings_path):
            self.class_embeddings = torch.load(
                self.embeddings_path,
                map_location=self.device,
                weights_only=True
            )
            print(f"[Identifier] Ready. {len(self.class_embeddings)} classes loaded.")
        else:
            print("[Identifier] No embeddings found. Please POST /train first.")

        if os.path.exists(self.counts_path):
            with open(self.counts_path, 'r') as f:
                self.class_counts = json.load(f)

    def get_transform(self) -> transforms.Compose:
        """Standard ImageNet preprocessing pipeline."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def identify(self, image_path: str) -> dict:
        """
        Identify the class of an image using cosine similarity search.

        Args:
            image_path: Path to the image file.

        Returns:
            {
                "success"    : True,
                "predicted"  : "class_name",
                "confidence" : 87.3,          # cosine similarity * 100
                "all_scores" : { "class_A": 87.3, "class_B": 61.2, ... }
            }
            or on failure:
            {
                "success": False,
                "error"  : "reason"
            }
        """
        if not self.class_embeddings:
            return {
                "success": False,
                "error":   "No embeddings loaded. Please POST /train first."
            }

        if not os.path.exists(image_path):
            return {
                "success": False,
                "error":   f"Image file not found: {image_path}"
            }

        try:
            image     = Image.open(image_path).convert("RGB")
            tensor    = self.get_transform()(image).unsqueeze(0).to(self.device)
            query_emb = get_embedding(self.backbone, tensor)  # (1, 1280)

            # Cosine similarity against every stored class mean embedding
            scores = {}
            for class_name, class_emb in self.class_embeddings.items():
                sim = torch.nn.functional.cosine_similarity(
                    query_emb, class_emb.to(self.device)
                ).item()
                scores[class_name] = round(sim * 100, 2)

            # Sort descending
            scores     = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
            best_class = next(iter(scores))
            best_score = scores[best_class]

            print(f"[Identifier] Predicted: '{best_class}' ({best_score}%)")

            return {
                "success":    True,
                "predicted":  best_class,
                "confidence": best_score,
                "all_scores": scores
            }

        except Exception as e:
            print(f"[Identifier] Error during identification: {e}")
            return {"success": False, "error": str(e)}

    def add_correction(self, image_path: str, correct_label: str) -> bool:
        """
        Instantly update the mean embedding for correct_label.
        Uses the running mean formula — no retraining, no index rebuild.

        Formula:
            new_mean = (old_mean * n + new_embedding) / (n + 1)

        Takes ~50ms on CPU. Persists changes to disk immediately.
        Also handles brand-new classes added dynamically.

        Args:
            image_path:    Path to the saved correction image.
            correct_label: The correct class name.

        Returns:
            True on success, False on error.
        """
        try:
            image   = Image.open(image_path).convert("RGB")
            tensor  = self.get_transform()(image).unsqueeze(0).to(self.device)
            new_emb = get_embedding(self.backbone, tensor).cpu()  # (1, 1280)

            if correct_label in self.class_embeddings:
                n        = self.class_counts.get(correct_label, 1)
                old_mean = self.class_embeddings[correct_label].cpu()
                new_mean = (old_mean * n + new_emb) / (n + 1)
                self.class_embeddings[correct_label] = new_mean
                self.class_counts[correct_label]     = n + 1
            else:
                # Dynamically register a brand-new class
                self.class_embeddings[correct_label] = new_emb
                self.class_counts[correct_label]     = 1
                print(f"[Identifier] New class registered dynamically: '{correct_label}'")

            # Persist to disk
            torch.save(self.class_embeddings, self.embeddings_path)
            with open(self.counts_path, 'w') as f:
                json.dump(self.class_counts, f, indent=2)

            print(
                f"[Identifier] Correction applied: '{correct_label}' "
                f"({self.class_counts[correct_label]} images in index)"
            )
            return True

        except Exception as e:
            print(f"[Identifier] add_correction error: {e}")
            return False
