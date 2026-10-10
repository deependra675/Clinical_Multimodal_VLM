# Clinical Multimodal VLM & Diagnostic Retrieval Engine

An end-to-end multimodal diagnostic pipeline aligning chest radiograph visual representations with clinical text findings in a shared 512-dimensional latent space using Microsoft's BiomedCLIP (PubMedBERT + ViT-B/16) and dense vector retrieval.

## Architecture Overview
1. **Visual & Text Alignment Tower:** Pretrained BiomedCLIP backbone extracting unit-normalized 512-D vectors for cross-modal similarity scoring.
2. **Dense Vector Index:** ChromaDB nearest-neighbor retrieval layer for historically verified reference case matching.
3. **Grounded Diagnostic Generation:** Multimodal Vision-Language Model conditioning to generate structured radiology reports (Findings & Impression).

## Setup & Execution
```bash
python -m venv .venv
# Activate environment
pip install -r requirements.txt
python embedder.py

## Features Implemented
- [x] **Dual-Encoder Latent Alignment (`embedder.py`):** 512-D normalized feature extraction across medical vision and clinical text domains using BiomedCLIP.
- [x] **Zero-Shot Diagnostic Screening (`zero_shot_classifier.py`):** Multi-label pathology screening via temperature-scaled cosine similarity over clinical prompts without task-specific training heads.
- [ ] **Multimodal Vector Index:** ChromaDB reference case indexing & nearest-neighbor retrieval.
- [ ] **Grounded VLM Reporting:** Structured findings generation conditioned on retrieved case context.