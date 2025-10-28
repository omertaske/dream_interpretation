# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from flask import Flask, jsonify, request, render_template, send_from_directory

# long_interpretation fonksiyonlarını içe aktar
import long_interpretation as li

app = Flask(__name__, static_folder="static", template_folder="templates")

# Veri setini ve index'i uygulama başlangıcında yükle
DATA_PATH = os.path.abspath(os.environ.get("RIYA_DATA", "_interpretations_contains_all.json"))
entries = li.load_dataset(DATA_PATH)
index = li.build_index(entries)
unique_words = sorted({e["word"] for e in entries})

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/interpret", methods=["POST"])
def api_interpret():
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Geçersiz JSON"}), 400

    text = (data or {}).get("text", "").strip()
    length = (data or {}).get("length", "long").strip().lower()
    top_k = (data or {}).get("top_k", 8)

    if not text:
        return jsonify({"error": "'text' alanı gerekli."}), 400
    if length not in {"short", "medium", "long"}:
        length = "long"
    try:
        top_k = int(top_k)
    except Exception:
        top_k = 8
    top_k = max(1, min(20, top_k))

    matches = li.find_matches(text, index)
    structured = li.build_long_interpretation_structured(text, matches, length=length, top_k=top_k)

    return jsonify({
        "ok": True,
        "input": {"text": text, "length": length, "top_k": top_k},
        "result": structured,
    })

@app.route("/api/symbols")
def api_symbols():
    q = request.args.get("q", "").strip()
    qf = li.fold_tr(q)
    out = []
    count = 0
    for w in unique_words:
        wf = li.fold_tr(w)
        if not qf or qf in wf:
            out.append(w)
            count += 1
            if count >= 20:
                break
    return jsonify({"ok": True, "items": out})

@app.route("/api/health")
def api_health():
    return jsonify({"ok": True})

@app.route("/robots.txt")
def robots():
    return send_from_directory(".", "robots.txt")

@app.route("/sitemap.xml")
def sitemap():
    return send_from_directory(".", "sitemap.xml")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = str(os.environ.get("DEBUG", "1")).lower() in {"1", "true", "yes"}
    # Reloader'ı devre dışı bırak ki arka planda tek proses kalsın
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
