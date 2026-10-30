"""
SDTD Multi-Agent Patent Intelligence Web Application
Production-ready Flask application for local execution and Render.com deployment.
"""
import os
import sys
import tempfile
import traceback
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename

# Ensure agents and modules are in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from agents.orchestrator import OrchestratorAgent

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload limit
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Lazy singleton for OrchestratorAgent
_orchestrator: Optional[OrchestratorAgent] = None

def get_orchestrator() -> OrchestratorAgent:
    global _orchestrator
    if _orchestrator is None:
        print("[Flask App] Initializing OrchestratorAgent...")
        _orchestrator = OrchestratorAgent()
    return _orchestrator


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "system": "SDTD Multi-Agent Patent Intelligence Engine",
        "faiss_ready": os.path.exists("patents_faiss.index"),
        "meta_ready": os.path.exists("patents_meta.parquet")
    })


@app.route("/api/search", methods=["POST"])
def search():
    """
    Unified search endpoint. Accepts either:
    1. Multipart/form-data with 'file', 'custom_ipc', 'top_k'
    2. JSON payload with 'query', 'custom_ipc', 'top_k'
    """
    try:
        orchestrator = get_orchestrator()
        custom_ipc = None
        top_k = 5
        query_val = None

        if request.content_type and "multipart/form-data" in request.content_type:
            # File Upload Form
            top_k = int(request.form.get("top_k", 5))
            ipc_str = request.form.get("custom_ipc", "")
            if ipc_str:
                custom_ipc = [c.strip().upper() for c in ipc_str.split(",") if c.strip()]

            if "file" in request.files and request.files["file"].filename:
                uploaded_file = request.files["file"]
                filename = secure_filename(uploaded_file.filename)
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                uploaded_file.save(save_path)
                query_val = save_path
            else:
                query_val = request.form.get("query", "").strip()
        else:
            # JSON Payload
            data = request.get_json(force=True, silent=True) or {}
            query_val = data.get("query", "").strip()
            top_k = int(data.get("top_k", 5))
            ipc_str = data.get("custom_ipc")
            if isinstance(ipc_str, list):
                custom_ipc = [c.strip().upper() for c in ipc_str if c.strip()]
            elif isinstance(ipc_str, str) and ipc_str.strip():
                custom_ipc = [c.strip().upper() for c in ipc_str.split(",") if c.strip()]

        if not query_val:
            return jsonify({"error": "Please provide either an uploaded document, a patent ID, or an invention query."}), 400

        res = orchestrator.run_prior_art_search(
            query=query_val,
            custom_ipc_codes=custom_ipc,
            top_k=top_k,
            save_report_path="prior_art_report.md"
        )

        return jsonify({
            "success": True,
            "target": res["target"],
            "candidates": res["candidates"],
            "elapsed_seconds": res["elapsed_seconds"],
            "report_markdown": res["report_markdown"]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/download_report")
def download_report():
    report_file = os.path.join(BASE_DIR, "prior_art_report.md")
    if os.path.exists(report_file):
        return send_file(report_file, as_attachment=True, download_name="prior_art_report.md", mimetype="text/markdown")
    return jsonify({"error": "No report generated yet."}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[*] Starting SDTD Patent Intelligence Server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
