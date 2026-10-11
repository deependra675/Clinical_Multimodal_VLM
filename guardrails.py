"""
guardrails.py
Contrastive anatomical guardrail for filtering non-thoracic radiographs and non-medical images.
"""
from typing import Dict, Any, List
import numpy as np
import torch

# Canonical anatomical anchors with clinical descriptions
ANATOMY_ANCHORS = {
    "chest": [
        "A frontal chest X-ray radiograph showing the lungs, heart, ribs, and thoracic cavity",
        "A posteroanterior or anteroposterior projection chest radiograph of the thorax",
    ],
    "extremity_bone": [
        "An X-ray radiograph of a leg, knee, femur, tibia, ankle, foot, arm, or hand",
        "A musculoskeletal extremity radiograph showing long bones and joints",
    ],
    "abdomen_pelvis": [
        "An abdominal X-ray radiograph showing the bowels, abdomen, and pelvis",
        "A plain radiograph of the abdomen and gastrointestinal tract",
    ],
    "non_medical": [
        "A standard photographic picture of a person, animal, nature, or everyday object",
        "A digital screenshot, document, or non-radiographic illustration",
    ],
}

# Minimum softmax probability required to certify the scan as thoracic
CHEST_CONFIDENCE_THRESHOLD = 0.50

# In-memory anchor embedding cache
_CACHED_ANATOMY_EMBEDDINGS: Dict[str, np.ndarray] = None


def _get_or_compute_anatomy_anchors(embedder) -> Dict[str, np.ndarray]:
    """Pre-computes and caches centroid text embeddings for each anatomical class."""
    global _CACHED_ANATOMY_EMBEDDINGS
    if _CACHED_ANATOMY_EMBEDDINGS is not None:
        return _CACHED_ANATOMY_EMBEDDINGS

    cached = {}
    for region, prompts in ANATOMY_ANCHORS.items():
        embeddings = []
        for prompt in prompts:
            # Handle different embedder signatures
            if hasattr(embedder, "embed_text"):
                emb = embedder.embed_text(prompt)
            elif hasattr(embedder, "embed_texts"):
                emb = embedder.embed_texts([prompt])[0]
            elif hasattr(embedder, "model") and hasattr(embedder, "tokenizer"):
                device = getattr(embedder, "device", "cpu")
                tokens = embedder.tokenizer([prompt]).to(device)
                with torch.no_grad():
                    features = embedder.model.encode_text(tokens)
                    features /= features.norm(dim=-1, keepdim=True)
                    emb = features.cpu().numpy()
            else:
                raise AttributeError("Unsupported embedder interface.")

            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            embeddings.append(np.squeeze(emb))

        # Mean centroid for the region, then L2-normalized
        centroid = np.mean(embeddings, axis=0)
        cached[region] = centroid / np.linalg.norm(centroid)

    _CACHED_ANATOMY_EMBEDDINGS = cached
    return _CACHED_ANATOMY_EMBEDDINGS


def check_chest_xray_validity(embedder, img_embedding) -> Dict[str, Any]:
    """
    Validates anatomical body region using competitive zero-shot classification.
    Rejects any scan where 'chest' is not the dominant anatomical class.
    """
    anchor_dict = _get_or_compute_anatomy_anchors(embedder)

    # 1. Normalize image embedding
    if isinstance(img_embedding, torch.Tensor):
        img_vec = img_embedding.detach().cpu().numpy().squeeze()
    else:
        img_vec = np.array(img_embedding).squeeze()

    norm = np.linalg.norm(img_vec)
    if norm > 0:
        img_vec = img_vec / norm

    # 2. Compute similarity against each anatomical region
    regions = list(anchor_dict.keys())
    similarities = np.array([np.dot(anchor_dict[r], img_vec) for r in regions])

    # 3. Softmax temperature scaling (standard temperature = 50.0)
    exp_sim = np.exp((similarities - np.max(similarities)) * 50.0)
    probs = exp_sim / np.sum(exp_sim)
    prob_map = {r: float(p) for r, p in zip(regions, probs)}

    predicted_region = regions[int(np.argmax(similarities))]
    chest_prob = prob_map["chest"]

    # 4. Enforce strict gate: Must be classified as 'chest' with adequate confidence
    is_valid = (predicted_region == "chest") and (chest_prob >= CHEST_CONFIDENCE_THRESHOLD)

    if is_valid:
        status_msg = "Scan verified as valid frontal chest radiograph (CXR)."
    else:
        human_readable_class = predicted_region.replace("_", " ").title()
        status_msg = (
            f"Scan rejected: Image classified as '{human_readable_class}' "
            f"(Chest confidence: {chest_prob * 100:.1f}%). "
            f"This engine only processes thoracic chest radiographs."
        )

    return {
        "is_valid_cxr": is_valid,
        "predicted_region": predicted_region,
        "chest_confidence": round(chest_prob, 4),
        "status": "APPROVED" if is_valid else "REJECTED_OOD",
        "message": status_msg,
        "anatomical_breakdown": {k: round(v * 100, 2) for k, v in prob_map.items()},
    }