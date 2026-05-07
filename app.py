import os
import tempfile
from pathlib import Path
import math
import pandas as pd

from flask import Flask, jsonify, request, send_from_directory, send_file

try:
    from flask_cors import CORS
except Exception:
    CORS = None

from agents.inference_agent import InferenceAgent
from config import MODELS_DIR, FEATURE_PIPELINE_PATH, TRAINING_METADATA_PATH

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
if CORS is not None:
    CORS(app)

inference_agent = InferenceAgent()


def _safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def _clean_single_result(result: dict) -> dict:
    counterfactual = result.get("counterfactual") or {}
    financial_impact = result.get("financial_impact") or {}
    final_verdict = result.get("final_verdict") or result.get("verdict")

    recommendation = {
        "FRAUD DETECTED": "Immediate block and analyst review",
        "HIGH RISK - MANUAL REVIEW": "Manual review required",
        "SUSPICIOUS - STEP-UP VERIFICATION": "Step-up verification required",
        "LEGITIMATE": "Allow transaction",
    }.get(final_verdict, result.get("recommendation") or "Manual review required")

    payload = {
        "FinalVerdict": final_verdict,
        "AdjustedRiskLevel": result.get("adjusted_risk_level") or result.get("risk_level"),
        "AdjustedRiskScore": round(
            _safe_float(
                result.get("adjusted_risk_score"),
                _safe_float(result.get("fraud_probability"))
            ),
            6
        ),
        "FraudProbability": round(_safe_float(result.get("fraud_probability")), 6),
        "Recommendation": recommendation,
        "EstimatedLoss_USD": round(
            _safe_float(financial_impact.get("estimated_loss_usd"), 0.0),
            2
        ),
        "ExecutiveSummary": result.get("executive_summary") or "",
        "LLM_Explanation": result.get("explanation") or "",
        "CounterfactualSummary": counterfactual.get("llm_explanation") or "",
        "Report": result.get("report") or "",
        "ModelName": result.get("model_name") or "Stacking Ensemble",
        "RiskAdjustment": result.get("risk_adjustment") or {},
        "VisibleContext": result.get("visible_context") or {},
    }
    return payload


def _clean_batch_result(result: dict) -> dict:
    output_csv = result["output_csv"]
    df = pd.read_csv(output_csv)

    # normalize NaN -> None
    df = df.where(pd.notnull(df), None)

    cleaned_rows = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()

        # support multiple possible backend column names
        estimated_loss = (
            row_dict.get("EstimatedLoss_USD")
            or row_dict.get("estimated_loss_usd")
            or row_dict.get("estimated_loss")
            or row_dict.get("financial_impact")
            or 0.0
        )

        llm_explanation = (
            row_dict.get("LLM_Explanation")
            or row_dict.get("llm_explanation")
            or row_dict.get("explanation")
            or "Not available"
        )

        counterfactual = (
            row_dict.get("CounterfactualSummary")
            or row_dict.get("CounterfactualExplanation")
            or row_dict.get("counterfactual_explanation")
            or row_dict.get("counterfactual")
            or "Not available"
        )

        cleaned_rows.append({
            "RowNo": idx + 1,

            # keep transaction identifier if present in CSV
            "TransactionID": (
                row_dict.get("TransactionID")
                or row_dict.get("transaction_id")
                or row_dict.get("Transaction_Id")
                or ""
            ),

            # show amount using whatever column exists
            "TransactionAmt": (
                row_dict.get("TransactionAmt")
                or row_dict.get("amount")
                or row_dict.get("transaction_amount")
                or 0
            ),

            "FinalVerdict": row_dict.get("FinalVerdict") or row_dict.get("final_verdict"),
            "AdjustedRiskLevel": row_dict.get("AdjustedRiskLevel") or row_dict.get("adjusted_risk_level"),
            "AdjustedRiskScore": _safe_float(
                row_dict.get("AdjustedRiskScore", row_dict.get("adjusted_risk_score"))
            ),
            "FraudProbability": _safe_float(
                row_dict.get("FraudProbability", row_dict.get("fraud_probability"))
            ),
            "EstimatedLoss_USD": round(_safe_float(
                row_dict.get("EstimatedLoss_USD")
                or row_dict.get("estimated_loss_usd")
                or row_dict.get("estimated_loss")
                or 0.0
            ), 2),
            "Recommendation": row_dict.get("Recommendation") or row_dict.get("recommendation"),
            "ExecutiveSummary": row_dict.get("ExecutiveSummary") or row_dict.get("executive_summary") or "",
            "LLM_Explanation": (
                row_dict.get("LLM_Explanation")
                or row_dict.get("llm_explanation")
                or row_dict.get("explanation")
                or ""
            ),
            "CounterfactualSummary": (
                row_dict.get("CounterfactualSummary")
                or row_dict.get("CounterfactualExplanation")
                or row_dict.get("counterfactual_explanation")
                or row_dict.get("counterfactual")
                or ""
            ),
        })

    summary = result.get("batch_summary") or {}

    payload = {
        "rows_scored": int(result.get("rows_scored", len(cleaned_rows))),
        "valid_rows": int(result.get("valid_rows", len(cleaned_rows))),
        "invalid_rows": int(result.get("invalid_rows", 0)),
        "errors": result.get("errors", []),
        "summary": {
            "total_rows": int(summary.get("total_rows", len(cleaned_rows))),
            "fraud_count": int(summary.get("fraud_count", 0)),
            "legitimate_count": int(summary.get("legitimate_count", 0)),
            "average_fraud_probability": _safe_float(
                summary.get("average_fraud_probability"), 0.0
            ),
            "average_adjusted_risk_score": _safe_float(
                summary.get("average_adjusted_risk_score"), 0.0
            ),
            "average_risk_level": summary.get("average_risk_level", "UNKNOWN"),
            "total_estimated_financial_impact_usd": _safe_float(
                summary.get("total_estimated_financial_impact_usd"), 0.0
            ),
            "risk_distribution": summary.get("risk_distribution", {}),
            "analyst_recommendation_summary": summary.get(
                "analyst_recommendation_summary", ""
            ),
            "executive_summary": summary.get("executive_summary", ""),
        },
        "rows": cleaned_rows,
        "output_csv": str(output_csv),
        "summary_json": str(result.get("summary_json", "")),
    }
    return payload


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, pd.Series):
        return _json_safe(obj.to_dict())
    elif isinstance(obj, pd.DataFrame):
        return _json_safe(obj.to_dict(orient="records"))
    else:
        return obj


@app.get("/health")
def health():
    artifacts = {
        "models_dir": MODELS_DIR,
        "feature_pipeline": FEATURE_PIPELINE_PATH,
        "training_metadata": TRAINING_METADATA_PATH,
    }
    artifact_status = {name: os.path.exists(path) for name, path in artifacts.items()}
    return jsonify(
        {
            "status": "ok" if all(artifact_status.values()) else "degraded",
            "service": "FraudSentinel API",
            "artifacts": artifact_status,
            "frontend": str(FRONTEND_DIR),
            "supported_endpoints": ["/predict-single", "/predict-csv", "/health"],
        }
    )


@app.post("/predict-single")
def predict_single():
    try:
        payload = request.get_json(silent=True) or {}
        result = inference_agent.predict_single(payload)
        cleaned = _json_safe(_clean_single_result(result))
        return jsonify(cleaned)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.post("/predict-csv")
def predict_csv():
    try:
        uploaded = request.files.get("file")
        if not uploaded:
            return jsonify({"error": "No CSV file uploaded. Use form-data key 'file'."}), 400

        mode = request.form.get("mode", "rich")

        suffix = Path(uploaded.filename or "input.csv").suffix or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            uploaded.save(tmp.name)
            temp_path = tmp.name

        try:
            result = inference_agent.predict_csv(temp_path, mode=mode)
            cleaned = _json_safe(_clean_batch_result(result))
            return jsonify(cleaned)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

@app.get("/download-output")
def download_output():
    try:
        file_path = request.args.get("path", "")
        if not file_path:
            return jsonify({"error": "Missing file path."}), 400

        file_path = os.path.abspath(file_path)

        if not os.path.exists(file_path):
            return jsonify({"error": "File not found."}), 404

        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:path>")
def serve_frontend(path):
    file_path = FRONTEND_DIR / path
    if file_path.exists() and file_path.is_file():
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)