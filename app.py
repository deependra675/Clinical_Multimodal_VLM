import asyncio
import io
import time
from contextlib import asynccontextmanager
from typing import Optional
import gradio as gr
from ui import demo as gradio_demo

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image

from pipeline import ClinicalDiagnosticEngine
from schemas import DiagnosticRespone, PrimaryScreening, ReferenceCase

# global pipeline container
engine: Optional[ClinicalDiagnosticEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("[Startup] Initializing Clinical Diagnostic Engine into memory...")
    engine = ClinicalDiagnosticEngine()
    print("[Startup] Engine warmed and ready for inference.")
    yield
    print("[Shutdown] Cleaning up Clinical pipeline resources...")
    engine = None

app = FastAPI(
    title="Clinical Multimodal Vision-Language Diagnostic API",
    description="Zero-shot chest radiograph pathology screening grounded with HNSW vector retrieval.",
    version="1.0.0",
    lifespan=lifespan,
)

def _run_inference_sync(
    image_bytes: bytes, top_k: int = 3
) -> dict:
    start_time = time.perf_counter()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Invalid image format: {str(exc)}")

    # single visual forward pass
    img_embedding = engine.embedder.embed_image(image)

    # zero shot screening
    probabilities = engine._compute_zero_shot_probabilities(img_embedding)
    primary_dx = max(probabilities.items(), key=lambda x: x[1])

    # vector retrieval for grounding evidence
    similar_cases = engine.vector_store.retrieve_similar_cases(
        img_embedding, top_k=top_k
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # format structured response
    reference_models = [
        ReferenceCase(
            case_id=c["case_id"],
            similarity=c["similarity"],
            pathology=c["metadata"].get("pathology", "Unknown"),
            impression=c["metadata"].get("impression", ""),
            findings=c["metadata"].get("findings", "")
        )
        for c in similar_cases
    ]

    return{
        "primary_screening": PrimaryScreening(
            detected_pathology=primary_dx[0],
            confidence_percent=primary_dx[1],
        ),
        "differential_distribution": probabilities,
        "retrieved_reference_cohort": reference_models,
        "processing_time_ms": round(elapsed_ms, 2),
    }

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "engine_loaded": engine is not None,
        "device": "CPU",
    }

@app.post(
    "/v1/diagnose",
    response_model=DiagnosticRespone,
    status_code=status.HTTP_200_OK,
    tags=["Clinical Inference"],
)
async def diagnostic_radiograph(
    file: UploadFile = File(
        ..., description="Chest radiograph image (PNG, JPEG, or DICOM-export)"
    ),
    top_k: int = Form(3, description="Number of nearest neighbors to retrieve"),
):
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Must be PNG or JPEG."
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )
    try:
        # offload cpu heavy computation to thread pool to prevent blocking the event loop
        result = await asyncio.to_thread(
            _run_inference_sync,
            image_bytes= image_bytes,
            top_k=top_k,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )
    return DiagnosticRespone(filename=file.filename or "unknown.png", **result)

app = gr.mount_gradio_app(app, gradio_demo, path="/dashboard")