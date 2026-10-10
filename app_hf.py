import gradio as gr
from PIL import Image
from pipeline import ClinicalDiagnosticEngine


print("[Hugging Face Space] Warming Clinical Multimodal Engine...")
engine = ClinicalDiagnosticEngine()
print("[Hugging Face Space] Engine ready.")


def run_clinical_diagnosis(image_input, top_k):
    if image_input is None:
        return "Please upload an image.", {}, []

    try:
        # Convert uploaded numpy array or PIL Image to RGB
        if not isinstance(image_input, Image.Image):
            image = Image.fromarray(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        # Multimodal forward pass
        img_embedding = engine.embedder.embed_image(image)
        probabilities = engine._compute_zero_shot_probabilities(img_embedding)
        primary_dx = max(probabilities.items(), key=lambda x: x[1])

        # Grounding retrieval
        similar_cases = engine.vector_store.retrieve_similar_cases(
            img_embedding, top_k=int(top_k)
        )

        summary_md = (
            f"### Primary Finding: **{primary_dx[0]}**\n\n"
            f"- **Confidence:** `{primary_dx[1]}%`\n"
        )

        prob_dict = {k: v / 100.0 for k, v in probabilities.items()}

        cohort_rows = [
            [
                c["case_id"],
                f"{c['similarity'] * 100:.1f}%",
                c["metadata"].get("pathology", "Unknown"),
                c["metadata"].get("impression", ""),
                c["metadata"].get("findings", ""),
            ]
            for c in similar_cases
        ]

        return summary_md, prob_dict, cohort_rows

    except Exception as exc:
        return f"Inference failed: {str(exc)}", {}, []


with gr.Blocks(title="Multimodal CXR Diagnostic Engine") as demo:
    gr.Markdown("# Clinical Multimodal Vision-Language Diagnostic Engine")
    gr.Markdown(
        "Zero-shot radiograph pathology screening grounded with **BiomedCLIP** "
        "feature extraction and **ChromaDB HNSW** vector retrieval."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Frontal Chest Radiograph (PNG / JPEG)")
            top_k_slider = gr.Slider(minimum=1, maximum=4, value=3, step=1, label="Reference Cases (top_k)")
            submit_btn = gr.Button("Run Diagnostic Screening", variant="primary")

        with gr.Column(scale=1):
            primary_output = gr.Markdown()
            distribution_output = gr.Label(num_top_classes=6, label="Differential Pathology Distribution")

    with gr.Row():
        cohort_table = gr.Dataframe(
            headers=["Case ID", "Visual Similarity", "Ground-Truth Diagnosis", "Radiologist Impression", "Findings"],
            datatype=["str", "str", "str", "str", "str"],
            label="Grounded Reference Cases (Case-Based Reasoning Evidence)",
            wrap=True,
        )

    submit_btn.click(
        fn=run_clinical_diagnosis,
        inputs=[input_image, top_k_slider],
        outputs=[primary_output, distribution_output, cohort_table],
    )

if __name__ == "__main__":
    demo.launch()