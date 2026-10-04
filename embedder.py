import torch
import open_clip
from PIL import Image
import numpy as np


class MedicalEmbedder:
    """Extracts normalized 512-dimensional semantic vectors from medical images and clinical text

    using Microsoft's BiomedCLIP architecture.
    """

    def __init__(self):
        print("Initializing BiomedCLIP model on CPU...")
        # Load Microsoft's BiomedCLIP model weights and image preprocessing pipeline
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.tokenizer = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.model.eval()
        print("BiomedCLIP initialized successfully.")

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Transforms a PIL image into a 512-dimensional normalized vector."""
        # Convert image into tensor and add batch dimension (1, 3, 224, 224)
        tensor = self.preprocess(image).unsqueeze(0)

        with torch.no_grad():
            features = self.model.encode_image(tensor)
            # Normalize vector to unit length (length = 1.0) so dot product equals cosine similarity
            features /= features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().squeeze(0)

    def embed_text(self, text: str) -> np.ndarray:
        """Transforms clinical text into a 512-dimensional normalized vector."""
        tokens = self.tokenizer([text], context_length=256)

        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().squeeze(0)

    def calculate_similarity(
        self, vec_a: np.ndarray, vec_b: np.ndarray
    ) -> float:
        """Computes cosine similarity between two normalized vectors (range: -1.0 to 1.0)."""
        return float(np.dot(vec_a, vec_b))


if __name__ == "__main__":
    embedder = MedicalEmbedder()

    # Create a synthetic 224x224 grayscale canvas representing a test radiograph
    synthetic_xray = Image.fromarray(
        np.full((224, 224), 128, dtype=np.uint8)
    ).convert("RGB")

    # Generate embeddings
    image_vector = embedder.embed_image(synthetic_xray)
    normal_text_vector = embedder.embed_text(
        "Normal chest radiograph with clear lungs."
    )
    disease_text_vector = embedder.embed_text(
        "Severe bilateral pneumonia and consolidation."
    )

    print("\n--- Output Verification ---")
    print(f"Image vector dimensions: {image_vector.shape} (Expected: (512,))")
    print(f"Text vector dimensions:  {normal_text_vector.shape}")
    print(
        f"Similarity (Image vs Normal text):  {embedder.calculate_similarity(image_vector, normal_text_vector):.4f}"
    )
    print(
        f"Similarity (Image vs Disease text): {embedder.calculate_similarity(image_vector, disease_text_vector):.4f}"
    )