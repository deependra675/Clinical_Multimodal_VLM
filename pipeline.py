import os
from typing import Any, Dict, List
import numpy as np
from PIL import Image
from embedder import MedicalEmbedder
from vector_store import ClinicalVectorStore

class ClinicalDiagnosticEngine:
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        print("Initializing Unified Diagnostic Engine...")
        self.embedder = MedicalEmbedder()
        self.vector_store = ClinicalVectorStore(
            persist_directory=persist_directory
        )

        # Standard diagnostic candidate labels
        self.candidate_labels = [
            "Normal chest radiograph with no acute cardiopulmonary abnormality",
            "Bacterial or viral pneumonia with focal consolidation or infiltrates",
            "Cardiomegaly with enlarged cardiac silhouette",
            "Pleural effusion with blunting of costophrenic angles",
            "Atelectasis with partial lung volume loss",
            "Pneumothorax with visible pleural line and absense of lung marking",
        ]

        self.pathology_keys = [
            "Normal",
            "Pneumonia",
            "Cardiomegaly",
            "Pleural Effusion",
            "Atelectasis",
            "Pneumothorax",
        ]

        # pre-compute and cache candidate text embeddings to save cpu cycles.
        print("Pre-computing text embeddings for diagnostic taxnomy...")
        self.text_embeddings = np.vstack(
            [self.embedder.embed_text(label) for label in self.candidate_labels]
        )
        print("Clinical Diagnostic Engine ready.")

    def _compute_zero_shot_probabilities(
            self, image_embedding: np.ndarray, temperature: float = 0.05
    ) -> Dict[str, float]:
        img_vec = image_embedding.squeeze() # Shape: (512,)

        # dot product in normalized space = cosine similarity
        similarities = np.dot(self.text_embeddings, img_vec) # shape: (num_labels,)

        # scaled dot product logits with temperature scaling
        scaled_logits = similarities / temperature

        # Numerically stable softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probabilities = exp_logits / np.sum(exp_logits)

        return {
            pathology: round(float(prob) * 100, 2)
            for pathology, prob in zip(self.pathology_keys, probabilities)
        }

    def analyze_scan(
            self, image_path: str, top_k_reference: int = 3
    ) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        image = Image.open(image_path).convert("RGB")

        # single visual forward pass
        img_embedding = self.embedder.embed_image(image)

        # zero shot screening
        probabilities = self._compute_zero_shot_probabilities(img_embedding)

        # primary suspected pathology (highest probabilities)
        primary_dx = max(probabilities.items(), key=lambda x: x[1])

        # vector retrieval for grounding evidence
        similar_cases = self.vector_store.retrieve_similar_cases(
            img_embedding, top_k=top_k_reference
        )
        return {
            "query_image": image_path,
            "primary_screening": {
                "detected_pathology": primary_dx[0],
                "confidence_percent": primary_dx[1],
            },
            "differential_distribution": probabilities,
            "retrieved_reference_cohort": similar_cases,
        }

if __name__ == "__main__":
    engine = ClinicalDiagnosticEngine()
    sample_scan = "data/sample_normal.png"
    if os.path.exists(sample_scan):
        print(f"\n--- Running Full Pipeline on: {sample_scan} ---")
        report = engine.analyze_scan(sample_scan, top_k_reference=3)

        print("\n[Diagnostic Screening Result]")
        print(
            f"Primary Finding: {report['primary_screening']['detected_pathology']}"
            f"({report['primary_screening']['confidence_percent']}%)"
        )
        print("\n[Differential Probability Distribution]")
        for pathology, prob in report["differential_distribution"].items():
            print(f"{pathology:<20}: {prob:>6.2f}%")

        print("\n[Top Grounded Reference Cases]")
        for i, ref in enumerate(report["retrieved_reference_cohort"], 1):
            print(
                f" {i}. Case {ref['case_id']} | Sim: {ref['similarity']*100:.1f}% |"
                f"Ground-Truth Dx: {ref['metadata']['pathology']}"
            )
            print(f" Impression: {ref['metadata']['impression']}")
    else:
        print(f"Sample scan not found at '{sample_scan}'.")