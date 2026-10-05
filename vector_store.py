import os
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
import numpy as np
from PIL import Image
from embedder import MedicalEmbedder

class ClinicalVectorStore:
    def __init__(
            self,
            persist_directory: str = "./data/chroma_db",
            collection_name: str = "chest_xray_reference_cohort",
    ):
        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # initialize persistent on disk client
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # use cosine distance space: d = 1 - (u . v)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def index_case(
            self, 
            case_id: str,
            embedding: np.ndarray,
            findings: str,
            impression: str,
            metadata: Dict[str, Any],
    ) -> None:
        # ensure 1 D float list format for ChromaDBw
        vec = embedding.squeeze().tolist()

        payload_metadata = {
            **metadata,
            "impression": impression,
        }

        self.collection.upsert(
            ids=[case_id],
            embeddings=[vec],
            documents=[findings],
            metadatas=[payload_metadata],
        )

    def retrieve_similar_cases(
            self, query_embedding: np.ndarray, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        vec = query_embedding.squeeze().tolist()

        results = self.collection.query(
            query_embeddings=[vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        if not results["ids"] or not results["ids"][0]:
            return retrieved

        for idx in range(len(results["ids"][0])):
            distance = results["distances"][0][idx]
            # convert cosine distance to cosine similarity: sim = 1 - distance
            similarity = 1.0 - distance

            retrieved.append(
                {
                    "case_id": results["ids"][0][idx],
                    "similarity": round(float(similarity), 4),
                    "findings": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                }
            )
        return retrieved

if __name__ == "__main__":
    print("--- Initializing Clinical Vector Store ---")
    embedder = MedicalEmbedder()
    store = ClinicalVectorStore()

    # Synthetic reference cohort representing confirmed clinical ground - truth cases
    reference_cases = [
        {
            "id": "CXR_NORM_001",
            "prompt": "frontal chest radiograph showing clear lungs and normal cardiothoracic ratio",
            "findings": "Lungs are clear bilaterally. No pleural effusion or pneumothorax identified.",
            "impressiopn": "Normal chest radiograph without acute cardiopulomnary findings.",
            "metadata": {"pathology": "Normal", "view": "PA", "patient_age": 42},
        },
        {
            "id": "CXR_PNEU_002",
            "prompt": "chest x-ray showing right lower lobe airspace consolidation in indicative of bacterial pneumonia",
            "findings": "Dense focal consolidation within the right lower lung base. Air bronchograms present.",
            "impressiopn": "cute right lower lobe pneumonia.",
            "metadata": {"pathology": "Pneumonia", "view": "PA", "patient_age": 58},
        },
        {
            "id": "CXR_CARD_003",
            "prompt": "chest radiograph demonstrating marked cardiomegaly and vascular redistribution",
            "findings": "Transverse cardiac diameter exceeds 50% of thoracic width. Prominent upper lobe vessels.",
            "impression": "Cardiomegaly with pulmonary venous congestion.",
            "metadata": {
                "pathology": "Cardiomegaly",
                "view": "PA",
                "patient_age": 67},
        },
        {
            "id": "CXR_EFFU_004",
            "prompt": "chest radiograph with blunting of the left costophrenic angle and meniscus sign",
            "findings": "Homogeneous opacity in the left lower hemithorax obscuring the hemidiaphragm.",
            "impression": "Moderate left-sided pleural effusion.",
            "metadata": {
                "pathology": "Pleural Effusion",
                "view": "PA",
                "patient_age": 71,
            },
        },
    ]

    print("\nIngesting refrence cohort embeddings into ChromaDBw...")
    for case in reference_cases:
        # cross-model initialization: generate reference vector from expert clinical descriptions
        vec = embedder.embed_text(case["prompt"])
        store.index_case(
            case_id=case["id"],
            embedding=vec,
            findings=case["findings"],
            impression=case["impression"],
            metadata=case["metadata"],
        )
        print(f"Indexed [{case['id']}] - {case['metadata']['pathology']}")

    print(f"\nTotal cases in collection: {store.collection.count()} reference records.")
    # retrieval test: query using the NIH PA normal scan
    test_scan_prompt = ("frontal chest radiograph, normal heart size, clear lung fields")
    query_vec = embedder.embed_text(test_scan_prompt)
    print(f"\nSearching nearest neighbors for query: '{test_scan_prompt}'...")
    matches = store.retrieve_similar_cases(query_vec, top_k=3)

    print("\n--- Top 3 Retrieved Clinical Matches ---")
    for rank, match in enumerate(matches, 1):
        print(f"Rank {rank}: Case ID={match['case_id']} | Cosine Sim={match['similarity'] * 100:.2f}%")
        print(f"    Dx: {match['metadata']['pathology']}")
        print(f"    Impression: {match['metadata']['impression']}")

    # Cross-modal visual search using raw pixel image
    image_path = "data/sample_normal.png"
    if os.path.exists(image_path):
        print(
            f"\nSearching database using raw image pixels: '{image_path}'..."
        )
        img = Image.open(image_path)
        img_vec = embedder.embed_image(img)

        visual_matches = store.retrieve_similar_cases(img_vec, top_k=3)

        print("\n--- Visual Query: Top Retrieved Case ---")
        top_match = visual_matches[0]
        print(f"Case ID: {top_match['case_id']}")
        print(f"Visual Similarity: {top_match['similarity'] * 100:.2f}%")
        print(f"Matched Diagnosis: {top_match['metadata']['pathology']}")
        print(f"Impression: {top_match['metadata']['impression']}")
        print(f"Clinical Findings: {top_match['findings']}")
    else:
        print(
            f"\nFile not found at '{image_path}'. Check your data folder path."
        )