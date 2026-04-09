"""
trainer.py - Embedding Index Builder

Replaces the full training pipeline with fast one-time embedding extraction.
No epochs, no gradients, no optimizer, no augmentation.

Each image is passed through the frozen MobileNetV2 backbone once.
The mean embedding across all images in a class becomes that class's
representative vector.

Performance on CPU:
  1000 classes x 25 images = 25,000 images
  Estimated index build time : 10-20 minutes  (one-time only)
  Per-correction update time : ~50ms          (instant, forever after)
"""

import os
import json
import time
import torch
from PIL import Image
from torchvision import transforms

from model import load_backbone, get_embedding, get_device


class Trainer:
    def __init__(self, data_dir: str, embeddings_path: str, counts_path: str):
        """
        Args:
            data_dir:        Path to folder of class subfolders (e.g. data/).
            embeddings_path: Where to save the mean embeddings (e.g. saved_model/embeddings.pt).
            counts_path:     Where to save image counts per class (e.g. saved_model/counts.json).
        """
        self.data_dir        = data_dir
        self.embeddings_path = embeddings_path
        self.counts_path     = counts_path
        self.device          = get_device()

    def get_transform(self) -> transforms.Compose:
        """Standard ImageNet preprocessing. No augmentation needed."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def validate_data(self):
        """
        Check data/ folder has at least 2 class subfolders each with at least 1 image.
        Only counts actual image files — ignores .DS_Store, README.txt, etc.
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data folder '{self.data_dir}' not found.")

        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}

        classes = [
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ]

        if len(classes) < 2:
            raise ValueError(f"Need at least 2 class folders, found: {classes}")

        total_images = 0
        for cls in classes:
            images = [
                f for f in os.listdir(os.path.join(self.data_dir, cls))
                if os.path.splitext(f)[1].lower() in valid_ext
            ]
            if len(images) == 0:
                raise ValueError(f"Class '{cls}' has no valid image files.")
            total_images += len(images)
            print(f"[Trainer] Class '{cls}': {len(images)} images")

        print(f"[Trainer] Total: {total_images} images across {len(classes)} classes")
        return classes, total_images

    def build_index(self) -> dict:
        """
        Extract and store the mean embedding for every class in data/.

        Steps:
          1. Load pretrained MobileNetV2 backbone (from cache, no internet needed).
          2. For every image: preprocess → forward pass → 1280-dim vector.
          3. Average all vectors per class → one mean embedding per class.
          4. Save embeddings.pt and counts.json to saved_model/.

        Returns:
            { class_name: image_count }  —  used for status reporting.
        """
        print("[Trainer] -- Building Embedding Index --")

        classes, total_images = self.validate_data()
        transform             = self.get_transform()
        valid_ext             = {'.jpg', '.jpeg', '.png', '.bmp'}

        backbone = load_backbone()
        backbone.to(self.device)
        backbone.eval()

        class_embeddings = {}
        class_counts     = {}
        start_time       = time.time()

        for i, class_name in enumerate(sorted(classes)):
            class_dir   = os.path.join(self.data_dir, class_name)
            image_files = [
                f for f in os.listdir(class_dir)
                if os.path.splitext(f)[1].lower() in valid_ext
            ]

            embeddings = []
            skipped    = 0

            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                try:
                    image  = Image.open(img_path).convert("RGB")
                    tensor = transform(image).unsqueeze(0).to(self.device)
                    emb    = get_embedding(backbone, tensor).cpu()  # (1, 1280)
                    embeddings.append(emb)
                except Exception as e:
                    print(f"[Trainer]   Skipped '{img_file}': {e}")
                    skipped += 1

            if not embeddings:
                print(f"[Trainer] WARNING: No valid images for '{class_name}', skipping.")
                continue

            mean_emb = torch.mean(torch.stack(embeddings), dim=0)  # (1, 1280)
            class_embeddings[class_name] = mean_emb
            class_counts[class_name]     = len(embeddings)

            elapsed = time.time() - start_time
            skip_note = f" ({skipped} skipped)" if skipped else ""
            print(
                f"[Trainer] [{i+1:>4}/{len(classes)}] '{class_name}': "
                f"{len(embeddings)} embeddings{skip_note} | {elapsed:.1f}s elapsed"
            )

        if not class_embeddings:
            raise ValueError("No embeddings were extracted. Check your data/ folder.")

        # Save to disk
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        torch.save(class_embeddings, self.embeddings_path)

        with open(self.counts_path, 'w') as f:
            json.dump(class_counts, f, indent=2)

        total_time = time.time() - start_time
        print(
            f"[Trainer] -- Index complete: "
            f"{len(class_embeddings)} classes, "
            f"{total_images} images, "
            f"{total_time:.1f}s total --"
        )
        return class_counts
