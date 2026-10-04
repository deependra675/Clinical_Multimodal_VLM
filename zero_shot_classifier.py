import numpy as np
import requests
import torch
from embedder import MedicalEmbedder
from PIL import Image
import io

class ZeroShotPathologyClassifier:
    def __init__(self, embedder: MedicalEmbedder):
        self.embedder = embedder

        # Standard clinicla evaluation labels
        self.pathology_labels = [
            "normal findings without acute cardiopulmonary abnormalities",
            "pneumonia with focal consolidation",
            "cardiomegaly with enlarged cardiac silhouette",
            "pleural effusion with blunted costophrenic angles",
            "pneumothorax with visible pleural line",
        ]

        # Clinical prompt templete to align with PubMedBERT pretraining
        self.prompt_template = "A chest radiograph displyaing {}."
        self. prompts = [
            self.prompt_template.format(label) for label in self.pathology_labels
        ]
        print("Pre-computiing text embeddings for clinical candidate labels...")

        # Pre-compute and stack text vectors: shape (num_classes, 512)
        self.text_embeddings = np.array(
            [self.embedder.embed_text(p) for p in self.prompts]
        )
        print("Candidate label embeddings ready.")

    def predict(
        self, image: Image.Image, temperature: float = 0.05
    ) -> dict[str, float]:
        img_vec = self.embedder.embed_image(image) # shape: (512,)

        # Dot product against normalized embeddings yields cosine similarities
        cosine_similarities = np.dot(self.text_embeddings, img_vec)

        # Scale by temperature and apply numerically stable softmax
        scaled_logits = cosine_similarities / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probabilities = exp_logits / np.sum(exp_logits)

        results = {
            label: float(prob)
            for label, prob in zip(self.pathology_labels, probabilities)
        }
        # Sort descending by probability
        return dict(
            sorted(results.items(), key=lambda item: item[1], reverse=True)
        )

if __name__ == "__main__":
    embedder = MedicalEmbedder()
    classifier = ZeroShotPathologyClassifier(embedder)

    # Standard open-access chest radiograph sample from NIH / Wikimedia commons
    sample_url = (
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    print("\nFetching sample chest radiograph...")
    response = requests.get(sample_url, headers=headers, timeout=15)
    response.raise_for_status()
    raw_img = Image.open(io.BytesIO(response.content)).convert("RGB")
    print(f"Sample image loaded successfully: {raw_img.size} px")

    print("\nRunning zero-shot diagnostic screening on test radiograph...")
    predictions = classifier.predict(raw_img)

    print("\n--- Diagnostic Probability Distribution ---")
    for condition, prob in predictions.items():
        print(f"{condition:<60} | {prob * 100:6.2f}%")
