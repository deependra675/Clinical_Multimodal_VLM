import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/v1/diagnose"


def run_clinical_diagnosis(image_filepath, top_k):
    if not image_filepath:
        return "⚠️ **Please upload a chest radiograph to proceed.**", {}, []

    try:
        with open(image_filepath, "rb") as f:
            files = {"file": (image_filepath, f, "image/png")}
            data = {"top_k": int(top_k)}
            res = requests.post(API_URL, files=files, data=data, timeout=30)

        # Handle Out-of-Distribution (OOD) rejections and validation errors cleanly
        if res.status_code == 422:
            detail_msg = res.json().get("detail", res.text)
            warning_md = (
                "### ⚠️ Input Rejected: Out-of-Distribution (OOD)\n"
                f"> **Status:** {detail_msg}\n\n"
                "Please verify the upload is an unobstructed frontal projection (PA/AP) chest radiograph."
            )
            return warning_md, {}, []

        if res.status_code != 200:
            return f"❌ **Server Error ({res.status_code}):** {res.text}", {}, []

        payload = res.json()
        primary = payload["primary_screening"]

        summary_md = (
            f"### Primary Finding: **{primary['detected_pathology']}**\n"
            f"- **Confidence:** `{primary['confidence_percent']}%`\n"
            f"- **Inference Latency:** `{payload['processing_time_ms']} ms`\n"
            f"- **Domain Status:** `Verified Chest Radiograph (CXR)`"
        )

        prob_dict = {
            k: v / 100.0 for k, v in payload["differential_distribution"].items()
        }

        cohort_rows = [
            [
                c["case_id"],
                f"{c['similarity'] * 100:.1f}%",
                c["pathology"],
                c["impression"],
                c["findings"],
            ]
            for c in payload["retrieved_reference_cohort"]
        ]

        return summary_md, prob_dict, cohort_rows

    except requests.exceptions.ConnectionError:
        return (
            "❌ **API Connection Failed:** Could not connect to `http://127.0.0.1:8000`. "
            "Ensure the FastAPI server is running (`uvicorn app:app --port 8000`).",
            {},
            [],
        )
    except Exception as exc:
        return f"❌ **Inference pipeline error:** {str(exc)}", {}, []


with gr.Blocks(title="Multimodal CXR Diagnostic Engine") as demo:
    gr.Markdown("# Clinical Multimodal Vision-Language Diagnostic Engine")
    gr.Markdown(
        "Zero-shot radiograph pathology screening grounded with **BiomedCLIP** feature extraction and **ChromaDB HNSW** vector retrieval."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="filepath", label="Frontal Chest Radiograph (PNG / JPEG)"
            )
            top_k_slider = gr.Slider(
                minimum=1, maximum=4, value=3, step=1, label="Reference Cases (top_k)"
            )
            submit_btn = gr.Button("Run Diagnostic Screening", variant="primary")

        with gr.Column(scale=1):
            primary_output = gr.Markdown(label="Diagnostic Assessment")
            distribution_output = gr.Label(
                num_top_classes=6, label="Differential Pathology Distribution"
            )

    with gr.Row():
        cohort_table = gr.DataFrame(
            headers=[
                "Case ID",
                "Visual Similarity",
                "Ground-Truth Diagnosis",
                "Radiologist Impression",
                "Findings",
            ],
            datatype=["str", "str", "str", "str", "str"],
            label="Grounded Reference Cases (Case-based reasoning evidence)",
            wrap=True,
        )

    submit_btn.click(
        fn=run_clinical_diagnosis,
        inputs=[input_image, top_k_slider],
        outputs=[primary_output, distribution_output, cohort_table],
    )

if __name__ == "__main__":
    demo.launch(share=True)