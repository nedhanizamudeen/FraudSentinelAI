# agents/feature_agent.py
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd

from tools.ml_tools import (
    clean_data,
    encode_categoricals,
    engineer_features,
    split_data,
    undersample_training_data,
    save_feature_pipeline,
)
from tools.data_tools import save_report
from tools.safe_llm import safe_llm_invoke
from config import GROQ_API_KEY, LLM_MODEL, REPORTS_DIR, FEATURE_PIPELINE_PATH


class FeatureAgent:
    def __init__(self):
        self.llm = ChatGroq(api_key=GROQ_API_KEY, model_name=LLM_MODEL, temperature=0)
        self.agent_name = "Feature Engineering Agent"

    def _think(self, feature_summary: str, eda_analysis: str) -> str:
        system_prompt = """You are the Feature Engineering Agent in FraudSentinel AI.
Respond in exactly 4 short bullet points. Each bullet = one sentence, max 20 words.
No headers, no paragraphs. Just 4 clean bullets starting with -"""
        user_message = f"""EDA findings: {eda_analysis[:300]}

Feature steps done:
{feature_summary[:500]}

Give exactly 4 bullet points:
- Overall quality of feature engineering
- Most useful new feature for fraud detection
- Any risk or concern
- One additional recommendation"""
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        return safe_llm_invoke(self.llm, messages, agent_name=self.agent_name, fallback_text="LLM unavailable")

    def run(self, state: dict) -> dict:
        print(f"\n{'='*60}")
        print(f"  [Feature]  {self.agent_name} — STARTING")
        print(f"{'='*60}")

        df = state["df"]
        columns_to_drop = state["columns_to_drop"]
        eda_analysis = state.get("eda_llm_analysis", "")
        all_summaries = []

        print("\n[Step 1/5] Cleaning data...")
        clean_result = clean_data(df, columns_to_drop)
        df_clean = clean_result["data"]
        all_summaries.append("=== DATA CLEANING ===\n" + clean_result["summary"])
        print(f"  {clean_result['summary']}")

        numeric_columns = [
            c for c in df_clean.select_dtypes(include=["number"]).columns.tolist()
            if c != "isFraud"
        ]
        categorical_columns = df_clean.select_dtypes(include=["object"]).columns.tolist()
        numeric_fill_values = df_clean[numeric_columns].median(numeric_only=True).to_dict() if numeric_columns else {}
        categorical_fill_values = {col: "Unknown" for col in categorical_columns}

        card1_mean_map = {}
        card1_count_map = {}
        addr1_mean_map = {}
        global_transaction_mean = 0.0
        if "TransactionAmt" in df_clean.columns:
            global_transaction_mean = float(pd.to_numeric(df_clean["TransactionAmt"], errors="coerce").fillna(0).mean())
        if {"TransactionAmt", "card1"}.issubset(df_clean.columns):
            card1_mean_map = df_clean.groupby("card1")["TransactionAmt"].mean().to_dict()
            card1_count_map = df_clean.groupby("card1")["card1"].count().to_dict()
        if {"TransactionAmt", "addr1"}.issubset(df_clean.columns):
            addr1_mean_map = df_clean.groupby("addr1")["TransactionAmt"].mean().to_dict()

        print("\n[Step 2/5] Engineering new features...")
        feature_result = engineer_features(df_clean.copy())
        df_featured = feature_result["data"]
        all_summaries.append("=== FEATURE ENGINEERING ===\n" + feature_result["summary"])
        print(f"  {feature_result['summary']}")

        print("\n[Step 3/5] Encoding categorical columns...")
        encode_result = encode_categoricals(df_featured)
        df_encoded = encode_result["data"]
        all_summaries.append("=== ENCODING ===\n" + encode_result["summary"])
        print(f"  {encode_result['summary']}")

        print("\n[Step 4/5] Splitting data into train/test sets...")
        split_result = split_data(df_encoded)
        all_summaries.append("=== DATA SPLIT ===\n" + split_result["summary"])
        print(f"  {split_result['summary']}")

        print("\n[Step 5/5] Undersampling only the training split...")
        under_result = undersample_training_data(split_result["X_train"], split_result["y_train"])
        X_train = under_result["X_train"]
        y_train = under_result["y_train"]
        all_summaries.append("=== TRAINING-ONLY UNDERSAMPLING ===\n" + under_result["summary"])
        print(f"  {under_result['summary']}")

        encoder_classes = {
            col: [str(x) for x in enc.classes_]
            for col, enc in encode_result["encoders"].items()
        }
        feature_names = split_result["feature_names"]
        feature_defaults = df_encoded[feature_names].median(numeric_only=True).to_dict()

        pipeline_artifact = {
            "pipeline_version": "2.2.0",
            "target_column": "isFraud",
            "raw_columns_before_cleaning": [c for c in df.columns if c != "isFraud"],
            "raw_input_columns_seen": list(df.columns),
            "columns_to_drop": columns_to_drop,
            "transaction_id_column": "TransactionID",
            "numeric_columns_before_encoding": numeric_columns,
            "categorical_columns_before_encoding": categorical_columns,
            "numeric_fill_values": numeric_fill_values,
            "categorical_fill_values": categorical_fill_values,
            "encoder_classes": encoder_classes,
            "feature_names": feature_names,
            "expected_feature_order": feature_names,
            "feature_default_values": feature_defaults,
            "new_features": feature_result["new_features"],
            "engineered_feature_rules": {
                "TransactionAmt_log": "log1p(TransactionAmt)",
                "TransactionAmt_to_card_mean": "TransactionAmt / (mean TransactionAmt by card1 + 1)",
                "transaction_hour": "(TransactionDT // 3600) % 24",
                "transaction_day": "(TransactionDT // 86400) % 7",
                "is_night_transaction": "1 if transaction_hour between 0 and 5 else 0",
                "card1_count": "count of card1 in training data",
                "amt_addr_ratio": "TransactionAmt / (mean TransactionAmt by addr1 + 1)",
            },
            "aggregate_maps": {
                "card1_transaction_mean": {str(k): float(v) for k, v in card1_mean_map.items()},
                "card1_count_map": {str(k): float(v) for k, v in card1_count_map.items()},
                "addr1_transaction_mean": {str(k): float(v) for k, v in addr1_mean_map.items()},
                "global_transaction_mean": float(global_transaction_mean),
            },
        }
        pipeline_path = save_feature_pipeline(pipeline_artifact, FEATURE_PIPELINE_PATH)
        all_summaries.append(f"=== FEATURE PIPELINE SAVED ===\nSaved reusable pipeline to: {pipeline_path}")

        print("\n[LLM] Feature Agent is reasoning about decisions...")
        combined_summary = "\n\n".join(all_summaries)
        llm_analysis = self._think(combined_summary, eda_analysis)
        short_lines = [l.strip() for l in llm_analysis.split("\n") if l.strip()][:6]
        print("\n  [LLM] LLM Decision:")
        for line in short_lines:
            print(f"     {line}")

        full_report = (
            f"FRAUDSENTINEL AI — FEATURE ENGINEERING AGENT REPORT\n"
            f"{'='*50}\n\n"
            f"{combined_summary}\n\n"
            f"{'='*50}\n"
            f"LLM REASONING:\n{llm_analysis}"
        )
        report_path = save_report("02_feature_report", full_report, REPORTS_DIR)
        print(f"  [Report] Report saved: {report_path}")
        print(f"\n  [OK] {self.agent_name} COMPLETE")

        return {
            **state,
            "df_processed": df_encoded,
            "X_train": X_train,
            "X_test": split_result["X_test"],
            "y_train": y_train,
            "y_test": split_result["y_test"],
            "feature_names": feature_names,
            "new_features": feature_result["new_features"],
            "encoders": encode_result["encoders"],
            "feature_summary": combined_summary,
            "feature_llm_analysis": llm_analysis,
            "feature_pipeline_path": pipeline_path,
            "status": "features_ready",
            "agent": self.agent_name,
        }