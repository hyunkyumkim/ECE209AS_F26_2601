"""Serve the local microphone UI and run pretrained YAMNet inference."""

import argparse
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import threading
import time
import traceback
from urllib.parse import parse_qs, urlparse

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
TARGETS = ("Whistling", "Knock", "Tap")


class Detector:
    def __init__(self, model_path=None):
        os.environ.setdefault("TFHUB_CACHE_DIR", str(ROOT / ".cache" / "yamnet"))
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        print("Loading pretrained YAMNet (first run may download the model)…", flush=True)
        import tensorflow_hub as hub

        self.model = hub.load(str(model_path) if model_path else MODEL_HANDLE)
        with open(self.model.class_map_path().numpy().decode(), encoding="utf-8") as label_file:
            self.labels = [row["display_name"] for row in csv.DictReader(label_file)]
        self.target_indices = {name: self.labels.index(name) for name in TARGETS}
        self.lock = threading.Lock()
        self.model(np.zeros(16000, dtype=np.float32))

    def predict(self, waveform, sample_rate):
        if sample_rate != 16000:
            divisor = math.gcd(sample_rate, 16000)
            waveform = resample_poly(waveform, 16000 // divisor, sample_rate // divisor)
        waveform = np.clip(waveform, -1, 1).astype(np.float32)

        started = time.perf_counter()
        with self.lock:
            scores, _, _ = self.model(waveform)
        # Peak patch scores preserve brief events better than whole-clip averaging.
        peak_scores = scores.numpy().max(axis=0)
        top_indices = np.argsort(peak_scores)[-5:][::-1]
        return {
            "scores": {
                name: float(peak_scores[index])
                for name, index in self.target_indices.items()
            },
            "top": [
                {"label": self.labels[index], "score": float(peak_scores[index])}
                for index in top_indices
            ],
            "rms": float(np.sqrt(np.mean(waveform ** 2))),
            "inference_ms": round((time.perf_counter() - started) * 1000),
        }


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.reply(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/health":
            self.reply(200, b'{"ready":true}')
        else:
            self.reply(404, b'{"error":"Not found"}')

    def do_POST(self):
        if urlparse(self.path).path != "/predict":
            self.reply(404, b'{"error":"Not found"}')
            return
        origin = self.headers.get("Origin")
        expected_origin = f"http://127.0.0.1:{self.server.server_port}"
        if origin and origin != expected_origin:
            self.reply(403, b'{"error":"Invalid origin"}')
            return

        try:
            parameters = parse_qs(urlparse(self.path).query)
            sample_rate = int(parameters["rate"][0])
            length = int(self.headers.get("Content-Length", "0"))
            if not 8000 <= sample_rate <= 192000:
                raise ValueError("Sample rate must be between 8000 and 192000 Hz")
            if not 4 <= length <= 1600000 or length % 4:
                raise ValueError("Invalid audio size")
            self.connection.settimeout(10)
            payload = self.rfile.read(length)
            if len(payload) != length:
                raise ValueError("Incomplete audio payload")
            waveform = np.frombuffer(payload, dtype="<f4").copy()
            if not np.isfinite(waveform).all():
                raise ValueError("Invalid audio samples")
            result = self.server.detector.predict(waveform, sample_rate)
            self.reply(200, json.dumps(result).encode())
        except (ValueError, KeyError, TimeoutError) as error:
            self.reply(400, json.dumps({"error": str(error)}).encode())
        except Exception:
            traceback.print_exc()
            self.reply(500, b'{"error":"Inference failed; check the server terminal"}')

    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="Local HTTP port (default: 8765)")
    parser.add_argument("--model-path", type=Path, help="Extracted local YAMNet SavedModel directory")
    parser.add_argument("--download-only", action="store_true", help="Cache and load the model, then exit")
    arguments = parser.parse_args()
    if not 1 <= arguments.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if arguments.model_path and not (arguments.model_path / "saved_model.pb").is_file():
        parser.error("--model-path must contain saved_model.pb")

    detector = Detector(arguments.model_path)
    if arguments.download_only:
        print("Model ready for offline inference.", flush=True)
        return

    server = ThreadingHTTPServer(("127.0.0.1", arguments.port), Handler)
    server.detector = detector
    print(f"Ready: http://127.0.0.1:{arguments.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
