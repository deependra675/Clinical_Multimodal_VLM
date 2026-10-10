import io
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image
from app import app, lifespan

@pytest.fixture(scope="session")
def synthetic_scan_bytes():
    buf = io.BytesIO()
    img = Image.new("RGB", (224, 223), color=(128, 128, 128))
    img.save(buf, format="PNG")
    return buf.getvalue()

@pytest.mark.asyncio
async def test_health_check():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["device"] == "CPU"

@pytest.mark.asyncio
async def test_diagnose_valid_image(synthetic_scan_bytes):
    async with lifespan(app):
        transport = ASGITransport(app=app)
        
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("synthetic_cxr.png", synthetic_scan_bytes, "image/png")}
            data = {"top_k": 2}
            res = await client.post("/v1/diagnose", files=files, data=data)

            print("Status code:", res.status_code)
            print("Response body:", res.text)
            assert res.status_code == 200
            payload = res.json()

            # contract assertion
            assert payload["filename"] == "synthetic_cxr.png"
            assert "detected_pathology" in payload["primary_screening"]
            assert 0.0 <= payload["primary_screening"]["confidence_percent"] <= 100.0
            assert len(payload["differential_distribution"]) == 6
            assert len(payload["retrieved_reference_cohort"]) == 2
            assert payload["processing_time_ms"] > 0

@pytest.mark.asyncio
async def test_diagnose_unsupported_media():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("notes.txt", b"Non-image text file", "text/plain")}
        res = await client.post("/v1/diagnose", files=files)
        assert res.status_code == 415

@pytest.mark.asyncio
async def test_diagnose_empty_file():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("empty.png", b"", "image/png")}
        res = await client.post("/v1/diagnose", files=files)
        assert res.status_code == 400