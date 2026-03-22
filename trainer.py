"""
trainer.py - Model Training & Retraining
CPU only mode.
Optimized for any dataset size:
  - Small  : ~20 images per class
  - Medium : ~100 images per class
  - Large  : 1000+ images per class
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import build_model, save_model, get_device


class Trainer:
    def __init__(self, data_dir: str, model_path: str, classes_file: str):
        self.data_dir     = data_dir
        self.model_path   = model_path
        self.classes_file = classes_file
        self.device       = get_device()

    def get_dataset_scale(self, total_images: int, num_classes: int) -> str:
        """
        Determine dataset scale based on total images.
          small  : avg < 50 images per class
          medium : avg 50-500 images per class
          large  : avg 500+ images per class
        """
        avg = total_images / max(num_classes, 1)
        if avg < 50:
            return "small"
        elif avg < 500:
            return "medium"
        else:
            return "large"

    def get_train_transform(self, scale: str) -> transforms.Compose:
        """
        Data augmentation pipeline.
        More aggressive augmentation for smaller datasets.
        """
        if scale == "small":
            # Heavy augmentation to compensate for few images
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif scale == "medium":
            # Moderate augmentation
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Large dataset - light augmentation, data speaks for itself
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def get_val_transform(self) -> transforms.Compose:
        """No augmentation for validation."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_training_config(self, scale: str, total_images: int) -> dict:
        """
        Auto-tune training hyperparameters based on dataset scale.

        Scale   | Epochs | Batch | LR     | Workers
        --------|--------|-------|--------|--------
        small   |   30   |   8   | 0.001  |   0
        medium  |   20   |  32   | 0.001  |   0
        large   |   10   |  64   | 0.0005 |   0
        """
        if scale == "small":
            return {
                "epochs"     : 30,
                "batch_size" : 8,
                "lr"         : 0.001,
                "num_workers": 0
            }
        elif scale == "medium":
            return {
                "epochs"     : 20,
                "batch_size" : 32,
                "lr"         : 0.001,
                "num_workers": 0
            }
        else:
            # Large: fewer epochs needed, larger batches for efficiency
            return {
                "epochs"     : 10,
                "batch_size" : 64,
                "lr"         : 0.0005,
                "num_workers": 0   # Keep 0 for Windows compatibility
            }

    def validate_data(self):
        """Check that the data folder has at least 2 classes with images."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data folder '{self.data_dir}' not found.")

        classes = [
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ]

        if len(classes) < 2:
            raise ValueError(f"Need at least 2 classes, but found: {classes}")

        total_images = 0
        for cls in classes:
            images = os.listdir(os.path.join(self.data_dir, cls))
            if len(images) == 0:
                raise ValueError(f"Class '{cls}' has no images!")
            total_images += len(images)
            print(f"[Trainer] Class '{cls}': {len(images)} images")

        print(f"[Trainer] Total images : {total_images}")
        print(f"[Trainer] Total classes: {len(classes)}")
        return classes, total_images

    def train(self) -> float:
        """
        Train the model on images inside data_dir.
        Auto-detects dataset scale and tunes hyperparameters accordingly.
        Returns best accuracy (0.0 - 100.0).
        """
        print("[Trainer] -- Starting Training --")
        print(f"[Trainer] Device: {self.device}")

        classes, total_images = self.validate_data()
        num_classes = len(classes)

        # Determine scale and config
        scale  = self.get_dataset_scale(total_images, num_classes)
        config = self.get_training_config(scale, total_images)

        print(f"[Trainer] Dataset scale : {scale.upper()}")
        print(f"[Trainer] Epochs        : {config['epochs']}")
        print(f"[Trainer] Batch size    : {config['batch_size']}")
        print(f"[Trainer] Learning rate : {config['lr']}")

        # Load dataset
        full_dataset = datasets.ImageFolder(
            root      = self.data_dir,
            transform = self.get_train_transform(scale)
        )

        # Save class names
        os.makedirs(os.path.dirname(self.classes_file), exist_ok=True)
        with open(self.classes_file, "w") as f:
            for cls in full_dataset.classes:
                f.write(cls + "\n")

        # Train/Val split (80/20) if enough images
        use_validation = total_images >= 20

        if use_validation:
            val_size   = max(1, int(0.2 * total_images))
            train_size = total_images - val_size
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )
            # Apply clean transform to val set
            val_full = datasets.ImageFolder(
                root      = self.data_dir,
                transform = self.get_val_transform()
            )
            # Use same indices for val
            from torch.utils.data import Subset
            val_indices = val_dataset.indices
            val_dataset = Subset(val_full, val_indices)

            print(f"[Trainer] Train set : {train_size} images")
            print(f"[Trainer] Val set   : {val_size} images")
        else:
            train_dataset = full_dataset
            val_dataset   = None

        train_loader = DataLoader(
            train_dataset,
            batch_size  = config["batch_size"],
            shuffle     = True,
            num_workers = config["num_workers"],
            pin_memory  = False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size  = config["batch_size"],
            shuffle     = False,
            num_workers = config["num_workers"],
            pin_memory  = False
        ) if val_dataset else None

        # Build model and optimizer
        model = build_model(num_classes)
        model.to(self.device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=config["lr"], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
        criterion = nn.CrossEntropyLoss()

        best_accuracy  = 0.0
        epochs         = config["epochs"]

        for epoch in range(epochs):
            epoch_start = time.time()

            # -- Training --
            model.train()
            train_loss    = 0.0
            train_correct = 0
            train_total   = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss    += loss.item()
                _, predicted   = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total   += labels.size(0)

                # Progress indicator for large datasets
                if scale == "large" and (batch_idx + 1) % 10 == 0:
                    print(
                        f"[Trainer]   Batch [{batch_idx+1}/{len(train_loader)}] "
                        f"Loss: {loss.item():.4f}",
                        end="\r"
                    )

            scheduler.step()
            train_accuracy = (100.0 * train_correct / train_total) if train_total > 0 else 0.0
            epoch_time     = time.time() - epoch_start

            # -- Validation --
            if val_loader:
                model.eval()
                val_correct = 0
                val_total   = 0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == labels).sum().item()
                        val_total   += labels.size(0)

                val_accuracy = (100.0 * val_correct / val_total) if val_total > 0 else 0.0

                print(
                    f"[Trainer] Epoch [{epoch+1:>3}/{epochs}] "
                    f"Loss: {train_loss:.4f} | "
                    f"Train: {train_accuracy:.1f}% | "
                    f"Val: {val_accuracy:.1f}% | "
                    f"Time: {epoch_time:.1f}s"
                )

                if val_accuracy >= best_accuracy:
                    best_accuracy = val_accuracy
                    save_model(model, self.model_path)
            else:
                print(
                    f"[Trainer] Epoch [{epoch+1:>3}/{epochs}] "
                    f"Loss: {train_loss:.4f} | "
                    f"Accuracy: {train_accuracy:.1f}% | "
                    f"Time: {epoch_time:.1f}s"
                )

                if train_accuracy >= best_accuracy:
                    best_accuracy = train_accuracy
                    save_model(model, self.model_path)

        print(f"[Trainer] -- Training Complete! Best Accuracy: {best_accuracy:.1f}% --")
        return best_accuracy
