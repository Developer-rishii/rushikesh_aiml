"""
A real, runnable HTTP service for the model, built on Python's stdlib
http.server (this sandbox has no network egress to install FastAPI/uvicorn,
so this gives an equally real, curl-able REST endpoint on localhost with
zero extra dependencies -- swap in FastAPI in prod without changing
model_service.py or the feature pipeline at all).

Endpoints:
  POST /predict  -> runs ModelService.predict(), returns score+explanation
  GET  /health   -> liveness + current model version
  GET  /metrics  -> current rolling MetricsStore snapshot (what a
                    Prometheus /metrics scrape would expose)
  POST /chaos    -> test-only hooks to inject latency/degenerate output/
                    unavailability (used by src/chaos/inject_failure.py)
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.serving.model_service import ModelService
from src.monitoring.metrics_store import MetricsStore
from src.monitoring.alerting import AlertEngine

metrics_store = MetricsStore(window=200)
model_service = ModelService(metrics_store=metrics_store)
alert_engine = AlertEngine()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # quiet; predictions.jsonl / alerts.log are the real audit trail

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {
                "status": "ok",
                "model_version": model_service.version,
                "feature_pipeline_version": model_service.metadata["feature_pipeline_version"],
            })
        elif self.path == "/metrics":
            snap = metrics_store.snapshot()
            fired = alert_engine.evaluate(snap)
            self._send(200, {"snapshot": snap, "alerts_fired_this_check": fired})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid json"})
            return

        if self.path == "/predict":
            result = model_service.predict(payload)
            self._send(200, result)
        elif self.path == "/chaos":
            action = payload.get("action")
            if action == "inject_latency":
                model_service.chaos_inject_latency(payload.get("ms", 300))
            elif action == "force_degenerate":
                model_service.chaos_force_degenerate_output(True)
            elif action == "force_unavailable":
                model_service.chaos_force_unavailable(True)
            elif action == "clear":
                model_service.clear_chaos()
            elif action == "freeze_reference":
                metrics_store.freeze_reference()
            else:
                self._send(400, {"error": f"unknown chaos action {action}"})
                return
            self._send(200, {"chaos_action_applied": action})
        else:
            self._send(404, {"error": "not found"})


def run(host="127.0.0.1", port=8899):
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"PlaceMux model service listening on http://{host}:{port} "
          f"(model_version={model_service.version})")
    server.serve_forever()


if __name__ == "__main__":
    run()
