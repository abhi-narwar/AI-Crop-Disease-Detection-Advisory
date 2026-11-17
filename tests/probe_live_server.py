"""
Probes live Uvicorn server running FastAPI backend on http://127.0.0.1:8000
"""

import sys
import time
import subprocess
import urllib.request
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
python_exe = sys.executable

print(f"Starting Uvicorn server process using {python_exe}...")
proc = subprocess.Popen(
    [python_exe, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
    cwd=str(PROJECT_ROOT),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

try:
    # Wait for server to become responsive
    max_wait = 15
    start_t = time.time()
    connected = False
    
    while time.time() - start_t < max_wait:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
                if response.status == 200:
                    connected = True
                    break
        except Exception:
            time.sleep(1)

    if not connected:
        stdout, stderr = proc.communicate(timeout=2)
        print("Uvicorn STDOUT:", stdout.decode(errors="ignore"))
        print("Uvicorn STDERR:", stderr.decode(errors="ignore"))
        raise RuntimeError("Server did not respond within timeout.")

    print(f"Server is live! (took {round(time.time() - start_t, 2)}s)")

    # 1. Probe health
    with urllib.request.urlopen("http://127.0.0.1:8000/health") as response:
        health_data = json.loads(response.read().decode())
        print("Live Server Health Response:", health_data)
        assert health_data["status"] == "healthy"

    # 2. Probe classes
    with urllib.request.urlopen("http://127.0.0.1:8000/classes") as response:
        classes_data = json.loads(response.read().decode())
        print(f"Live Server Total Classes: {classes_data['total_classes']}, Crops: {classes_data['crops']}")
        assert classes_data["total_classes"] == 15

    # 3. Probe root / (Frontend HTML)
    with urllib.request.urlopen("http://127.0.0.1:8000/") as response:
        html = response.read().decode()
        print(f"Live Server Root HTML Response Length: {len(html)} bytes")
        assert "<title>AI Crop Disease Detection & Advisory System</title>" in html

    print("\n✓ ALL LIVE SERVER PROBES COMPLETED SUCCESSFULLY!")
finally:
    print("Terminating server process...")
    proc.terminate()
    proc.wait()
    print("Server stopped.")
