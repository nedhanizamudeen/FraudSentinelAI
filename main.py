# main.py
import json
import os
import sys
import time
import warnings
import hashlib

warnings.filterwarnings("ignore")


REQUIRED_MODEL_ARTIFACTS = [
    "best_model.pkl",
    "xgboost_model.pkl",
    "lightgbm_model.pkl",
    "catboost_model.pkl",
    "stacking_meta_learner.pkl",
]


def print_banner():
    print("""
======================================================================
  FRAUDSENTINEL AI v2.2 BACKEND
  Backend-only training and inference package
  Modes: pipeline | predict-single | predict-csv
======================================================================
""")


def check_setup(mode: str) -> bool:
    from config import GROQ_API_KEY, TRANSACTION_FILE, IDENTITY_FILE, FEATURE_PIPELINE_PATH, MODELS_DIR, TRAINING_METADATA_PATH

    all_good = True
    if GROQ_API_KEY == "paste_your_groq_key_here":
        print("  [WARN] GROQ_API_KEY not set. LLM outputs will fall back to safe template text.")
    else:
        print("  [OK] Groq API key found")

    if mode == "--pipeline":
        for path, label in [(TRANSACTION_FILE, "train_transaction.csv"), (IDENTITY_FILE, "train_identity.csv")]:
            if not os.path.exists(path):
                print(f"  [ERROR] {label} not found: {path}")
                all_good = False
            else:
                print(f"  [OK] {label} found")
    else:
        if not os.path.exists(MODELS_DIR):
            print("  [ERROR] models/ directory not found. Run --pipeline first.")
            all_good = False
        for artifact in REQUIRED_MODEL_ARTIFACTS:
            artifact_path = os.path.join(MODELS_DIR, artifact)
            if not os.path.exists(artifact_path):
                print(f"  [ERROR] Missing model artifact: {artifact_path}")
                all_good = False
        for path, label in [
            (FEATURE_PIPELINE_PATH, "feature_pipeline.pkl"),
            (TRAINING_METADATA_PATH, "training_metadata.json"),
        ]:
            if not os.path.exists(path):
                print(f"  [ERROR] Missing inference artifact: {path}")
                all_good = False
    return all_good


def save_training_metadata(final_state: dict):
    from config import TRAINING_METADATA_PATH, FEATURE_PIPELINE_PATH

    feature_names = final_state.get("feature_names", []) or []
    trained_model_names = list((final_state.get("trained_models") or {}).keys())
    best_model_name = final_state.get("best_model_name") or "Stacking Ensemble"

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trained_model_names": trained_model_names,
        "best_model": best_model_name,
        "feature_count": len(feature_names),
        "feature_hash": hashlib.sha256("|".join(feature_names).encode("utf-8")).hexdigest() if feature_names else None,
        "feature_pipeline_path": FEATURE_PIPELINE_PATH,
    }

    os.makedirs(os.path.dirname(TRAINING_METADATA_PATH), exist_ok=True)
    with open(TRAINING_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return TRAINING_METADATA_PATH


def verify_training_artifacts(final_state: dict):
    from config import MODELS_DIR, FEATURE_PIPELINE_PATH, TRAINING_METADATA_PATH

    missing = []
    for artifact in REQUIRED_MODEL_ARTIFACTS:
        if not os.path.exists(os.path.join(MODELS_DIR, artifact)):
            missing.append(artifact)
    if not os.path.exists(FEATURE_PIPELINE_PATH):
        missing.append("feature_pipeline.pkl")
    if not os.path.exists(TRAINING_METADATA_PATH):
        missing.append("training_metadata.json")

    if missing:
        raise RuntimeError(
            "Pipeline completed but required artifacts are missing: " + ", ".join(missing)
        )

    if not final_state.get("feature_names"):
        raise RuntimeError("Pipeline completed without feature_names in final state.")


def run_full_pipeline():
    from orchestrator.graph import run_pipeline

    print("\n  Starting full pipeline...\n")
    start_time = time.time()
    final_state = run_pipeline(training_only=True)
    elapsed = time.time() - start_time
    mins, secs = int(elapsed // 60), int(elapsed % 60)
    print(f"\n  Pipeline completed in {mins}m {secs}s")
    if "best_auc" in final_state:
        print(f"  Best Model : {final_state.get('best_model_name', 'N/A')}")
        print(f"  Best AUC   : {final_state.get('best_auc', 0):.4f}")

    metadata_path = save_training_metadata(final_state)
    print(f"  Training metadata saved: {metadata_path}")
    verify_training_artifacts(final_state)
    print("  [OK] All required training artifacts are available for inference")
    return final_state


def run_predict_single(json_path: str):
    from agents.inference_agent import InferenceAgent

    with open(json_path, "r", encoding="utf-8") as f:
        transaction = json.load(f)
    result = InferenceAgent().predict_single(transaction)
    print(result["report"])
    if result.get("counterfactual"):
        print("\nCounterfactual summary:")
        print(result["counterfactual"].get("llm_explanation", ""))


def run_predict_csv(csv_path: str):
    from agents.inference_agent import InferenceAgent

    result = InferenceAgent().predict_csv(csv_path)
    print(f"\nScored {result['rows_scored']:,} rows")
    print(f"Saved batch predictions to: {result['output_csv']}")


def main():
    print_banner()
    mode = sys.argv[1] if len(sys.argv) > 1 else "--pipeline"
    print(f"\n  Mode: {mode}")
    print("\n  Checking setup...")
    if not check_setup(mode):
        print("\n  Setup incomplete. Please fix the issues above and try again.\n")
        sys.exit(1)
    print("\n  All checks passed!\n")

    if mode == "--pipeline":
        run_full_pipeline()
    elif mode == "--predict-single":
        if len(sys.argv) < 3:
            print("Usage: python main.py --predict-single path/to/transaction.json")
            sys.exit(1)
        run_predict_single(sys.argv[2])
    elif mode == "--predict-csv":
        if len(sys.argv) < 3:
            print("Usage: python main.py --predict-csv path/to/input.csv")
            sys.exit(1)
        run_predict_csv(sys.argv[2])
    else:
        print("Usage: python main.py [--pipeline | --predict-single path.json | --predict-csv path.csv]")
        sys.exit(1)


if __name__ == "__main__":
    main()