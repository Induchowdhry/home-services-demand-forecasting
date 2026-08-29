"""Flask API skeleton for the 7-day demand forecast."""
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.get("/forecast")
def forecast():
    city = request.args.get("city")
    service_type = request.args.get("service_type")
    if not city or not service_type:
        return jsonify({"error": "city and service_type are required"}), 400
    return jsonify({
        "city": city,
        "service_type": service_type,
        "forecast": [],
        "status": "model not loaded; awaiting correct booking dataset"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
