"""
identifier.py - Image Identification (Inference)

Loads the trained model from disk and runs prediction on input images.
Supports hot-reload after retraining without restarting the server.
"""

import os
import torch
from torchvision import transforms
from PIL import Image

from model import load_model


class Identifier:
    def __init__(self, model_path: str, classes_file: str):
        """
        Args:
            model_path:   Path to the saved model weights (.pth).
            classes_file: Path to the saved class names (.txt).
        """
        self.model_path   = model_path
        self.classes_file = classes_file
        self.model        = None
        self.class_names  = []
        self.device       = torch.device("cpu")  # Windows CPU only
        self.reload()

    def reload(self):
        """
        Reload model and class names from disk.
        Called automatically after training or retraining completes.
        """
        self.model, self.class_names = load_model(self.model_path, self.classes_file)

        if self.model:
            print(f"[Identifier] Ready. Classes: {self.class_names}")
        else:
            print("[Identifier] No model loaded. Please POST /train first.")

    def get_transform(self) -> transforms.Compose:
        """Preprocessing pipeline for inference (no augmentation)."""
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
        Identify the thing in an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            {
                "success"    : True,
                "predicted"  : "thing_name",
                "confidence" : 94.5,
                "all_scores" : { "thing_A": 94.5, "thing_B": 5.5 }
            }
            or on failure:
            {
                "success": False,
                "error"  : "reason"
            }
        """
        # Guard: model not trained yet
        if self.model is None:
            return {
                "success": False,
                "error":   "Model not trained yet. Please POST /train first."
            }

        # Guard: image file exists
        if not os.path.exists(image_path):
            return {
                "success": False,
                "error":   f"Image file not found: {image_path}"
            }

        try:
            # Load and preprocess image
            image  = Image.open(image_path).convert("RGB")
            tensor = self.get_transform()(image).unsqueeze(0).to(self.device)

            # Run inference
            self.model.eval()
            with torch.no_grad():
                outputs       = self.model(tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            # Top prediction
            confidence, pred_idx = torch.max(probabilities, 0)
            predicted_class      = self.class_names[pred_idx.item()]
            confidence_pct       = round(confidence.item() * 100, 2)

            # All class scores (sorted descending)
            all_scores = {
                self.class_names[i]: round(probabilities[i].item() * 100, 2)
                for i in range(len(self.class_names))
            }
            all_scores = dict(
                sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            )

            print(f"[Identifier] Predicted: '{predicted_class}' ({confidence_pct}%)")

            return {
                "success":    True,
                "predicted":  predicted_class,
                "confidence": confidence_pct,
                "all_scores": all_scores
            }

        except Exception as e:
            print(f"[Identifier] Error during identification: {e}")
            return {
                "success": False,
                "error":   str(e)
            }
