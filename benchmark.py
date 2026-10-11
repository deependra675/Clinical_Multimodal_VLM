import time
import io
import os
import statistics
import requests
from PIL import Image

API_URL = "http://localhost:8000/v1/diagnose"
HEALTH_URL = "http://localhost:8000/health"
NUM_ITERATIONS = 25
WARMUP_ITERATIONS = 2
REQUEST_TIMEOUT = 60

def load_test_image_bytes() -> bytes:
    target_file = "sample_cxr.jpg"
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"Missing {target_file}. Download or place a frontal CXR in the root directory.")
    print(f"Loading Clinical test timage: {target_file}")
    with open(target_file, "rb") as f:
        return f.read()
           
def run_benchmark():
    try:
        health_resp = requests.get(HEALTH_URL, timeout=10)
        health_resp.raise_for_status()
        print("API Healthcheck: PASSED")
    except Exception as e:
        print(f"Healthcheck failed: {e}")
        return

    img_bytes = load_test_image_bytes()

    print(f"\nRunning {WARMUP_ITERATIONS} warm-up cycles...")
    for _ in range(WARMUP_ITERATIONS):
        files = {"file": ("cxr.png", io.BytesIO(img_bytes), "image/png")}
        requests.post(API_URL, files=files, timeout=REQUEST_TIMEOUT)

    print(f"Profiling {NUM_ITERATIONS} inference iterations...")
    latencies = []
    statuses = []

    t_start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        files = {"file": ("cxr.png", io.BytesIO(img_bytes), "image/png")}
        t0 = time.perf_counter()
        resp = requests.post(API_URL, files=files, timeout=REQUEST_TIMEOUT)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code in [200, 422]:
            latencies.append(elapsed_ms)
            statuses.append(resp.status_code)
            print(f"  [{i+1:02d}/{NUM_ITERATIONS}] {elapsed_ms:.1f} ms (Status {resp.status_code})")
        else:
            print(f"  [{i+1:02d}] Unexpected error: {resp.status_code} - {resp.text}")

    total_wall_time = time.perf_counter() - t_start

    if not latencies:
        print("No valid runs recorded.")
        return

    latencies.sort()
    p50 = statistics.median(latencies)
    p90 = latencies[int(len(latencies) * 0.90)]
    p95 = latencies[int(len(latencies) * 0.95)]
    mean_lat = statistics.mean(latencies)
    throughput = len(latencies) / total_wall_time
    primary_status = "200 (Full Inference + Retrieval)" if 200 in statuses else "422 (OOD Guardrail Filter)"

    print("\n" + "=" * 50)
    print("       CLINICAL VLM BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Evaluated Pipeline  : {primary_status}")
    print(f"Completed Runs      : {len(latencies)} / {NUM_ITERATIONS}")
    print(f"Mean Latency        : {mean_lat:.2f} ms")
    print(f"Median (P50) Latency: {p50:.2f} ms")
    print(f"P90 Latency         : {p90:.2f} ms")
    print(f"P95 Latency         : {p95:.2f} ms")
    print(f"Throughput          : {throughput:.2f} scans/sec")
    print("=" * 50)


if __name__ == "__main__":
    run_benchmark()