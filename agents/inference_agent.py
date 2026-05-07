# agents/inference_agent.py

import json
import os
from datetime import datetime

import joblib
import pandas as pd

from agents.decision_agent import DecisionAgent
from agents.counterfactual_agent import CounterfactualAgent
from config import (
    MODELS_DIR,
    FEATURE_PIPELINE_PATH,
    TRAINING_METADATA_PATH,
    INFERENCE_OUTPUT_DIR,
    SINGLE_TRANSACTION_INPUT_SCHEMA,
)
from tools.ml_tools import (
    load_feature_pipeline,
    preprocess_for_inference,
    predict_with_stack,
    validate_required_artifacts,
    validate_artifact_compatibility,
    map_basic_transaction_input,
    extract_inference_risk_signals,
    adjust_probability_with_signals,
)
from tools.ml_tools import filter_user_visible_context

class InferenceAgent:
    def __init__(self):
        self.agent_name = "Inference Agent"
        self.decision_agent = DecisionAgent()
        self.cf_agent = CounterfactualAgent()
        self.risk_order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        self.inverse_risk_order = {1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}

    def _load_artifacts(self) -> dict:
        artifact_check = validate_required_artifacts(
            MODELS_DIR, FEATURE_PIPELINE_PATH, TRAINING_METADATA_PATH
        )
        required = artifact_check["paths"]

        artifacts = {
            "meta_learner": joblib.load(required["stacking_meta_learner.pkl"]),
            "xgb_model": joblib.load(required["xgboost_model.pkl"]),
            "lgb_model": joblib.load(required["lightgbm_model.pkl"]),
            "cat_model": joblib.load(required["catboost_model.pkl"]),
            "feature_pipeline": load_feature_pipeline(required["feature_pipeline.pkl"]),
            "metadata": None,
        }

        if artifact_check["metadata_exists"]:
            with open(artifact_check["metadata_path"], "r", encoding="utf-8") as f:
                artifacts["metadata"] = json.load(f)

        validate_artifact_compatibility(
            artifacts["feature_pipeline"],
            {
                "xgb_model": artifacts["xgb_model"],
                "lgb_model": artifacts["lgb_model"],
                "cat_model": artifacts["cat_model"],
                "meta_learner": artifacts["meta_learner"],
            },
            artifacts["metadata"],
        )
        return artifacts

    def _risk_impact_label(self, risk_level: str, estimated_loss: float) -> str:
        if risk_level == "CRITICAL":
            return f"Critical exposure — immediate block/review recommended (${estimated_loss:,.2f} at stake)"
        if risk_level == "HIGH":
            return f"High exposure — analyst review recommended (${estimated_loss:,.2f} potentially protected)"
        if risk_level == "MEDIUM":
            return f"Moderate exposure — monitor or step-up verification (${estimated_loss:,.2f} transaction value)"
        return f"Low exposure — allow with routine monitoring (${estimated_loss:,.2f} transaction value)"

    def _normalize_csv_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        aliases = {
            str(k).lower(): v
            for k, v in SINGLE_TRANSACTION_INPUT_SCHEMA.get("aliases", {}).items()
        }
        renamed = {}
        for col in df.columns:
            canonical = aliases.get(str(col).strip().lower())
            if canonical and canonical not in df.columns:
                renamed[col] = canonical
        return df.rename(columns=renamed)

    def _validate_csv_input(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        if df is None or df.empty:
            raise ValueError(
                "CSV input is empty. Provide a non-empty CSV with at least one transaction row."
            )

        df = self._normalize_csv_columns(df.copy())

        recognized = set(
            SINGLE_TRANSACTION_INPUT_SCHEMA.get("required", [])
            + SINGLE_TRANSACTION_INPUT_SCHEMA.get("optional", [])
        )
        present = [c for c in df.columns if c in recognized]
        if not present:
            raise ValueError(
                "CSV schema not recognized. Include at least one supported column such as "
                "TransactionAmt, TransactionDT, ProductCD, card1, card4, addr1, "
                "P_emaildomain, R_emaildomain, DeviceType."
            )

        critical_any = SINGLE_TRANSACTION_INPUT_SCHEMA.get(
            "critical_csv_any_of", ["TransactionAmt"]
        )
        if not any(c in df.columns for c in critical_any):
            raise ValueError(
                f"CSV missing critical column(s). Need one of: {', '.join(critical_any)}."
            )

        numeric_candidates = [
            "TransactionAmt",
            "TransactionDT",
            "card1",
            "card2",
            "card3",
            "card5",
            "addr1",
            "addr2",
            "dist1",
        ]
        for col in [c for c in numeric_candidates if c in df.columns]:
            coerced = pd.to_numeric(df[col], errors="coerce")
            bad_count = int(coerced.isna().sum() - df[col].isna().sum())
            if bad_count > 0:
                raise ValueError(
                    f"CSV column '{col}' contains {bad_count} invalid numeric value(s)."
                )
            df[col] = coerced

        if mode not in {"fast", "rich"}:
            raise ValueError("CSV inference mode must be 'fast' or 'rich'.")

        return df

    def _apply_risk_adjustment(
        self, raw_row: dict, base_probability: float, feature_pipeline: dict
    ) -> dict:
        signals = extract_inference_risk_signals(raw_row, feature_pipeline)
        adjusted = adjust_probability_with_signals(base_probability, signals)
        return adjusted

    def _aggregate_batch_summary(self, out: pd.DataFrame) -> dict:
        total = int(len(out))
        fraud_rows = out[out["PredictedLabel"] == 1]
        legit_rows = out[out["PredictedLabel"] == 0]

        avg_prob = round(float(out["FraudProbability"].mean()), 6) if total else 0.0
        avg_adjusted = (
            round(float(out["AdjustedRiskScore"].mean()), 6)
            if total and "AdjustedRiskScore" in out.columns
            else avg_prob
        )
        total_impact = round(
            float(pd.to_numeric(out.get("EstimatedLoss_USD", 0), errors="coerce").fillna(0).sum()),
            2,
        )

        if "AdjustedRiskLevel" in out.columns:
            avg_risk_level = out["AdjustedRiskLevel"].value_counts().idxmax()
            risk_distribution = out["AdjustedRiskLevel"].value_counts().to_dict()
        elif "RiskLevel" in out.columns:
            avg_risk_level = out["RiskLevel"].value_counts().idxmax()
            risk_distribution = out["RiskLevel"].value_counts().to_dict()
        else:
            avg_risk_level = "UNKNOWN"
            risk_distribution = {}

        recommendation_counts = (
            out.get("Recommendation", pd.Series(dtype=str)).value_counts().to_dict()
        )
        recommendation_summary = (
            "; ".join(f"{k}: {v}" for k, v in recommendation_counts.items())
            if recommendation_counts
            else "No analyst recommendations generated"
        )

        return {
            "total_rows": total,
            "fraud_count": int(len(fraud_rows)),
            "legitimate_count": int(len(legit_rows)),
            "average_fraud_probability": avg_prob,
            "average_adjusted_risk_score": avg_adjusted,
            "average_risk_level": avg_risk_level,
            "total_estimated_financial_impact_usd": total_impact,
            "risk_distribution": risk_distribution,
            "analyst_recommendation_summary": recommendation_summary,
            "executive_summary": (
                f"FraudSentinel scored {total:,} rows, flagging {len(fraud_rows):,} as fraud and "
                f"{len(legit_rows):,} as legitimate. Average model fraud probability was {avg_prob*100:.1f}%, "
                f"while average adjusted risk score was {avg_adjusted*100:.1f}% with overall risk centered at {avg_risk_level}. "
                f"Total estimated financial impact was ${total_impact:,.2f}; recommendation mix: {recommendation_summary}."
            ),
        }

    def _build_single_transaction_report(
        self,
        transaction: dict,
        model_name: str,
        base_probability: float,
        base_risk_level: str,
        base_model_verdict: str,
        estimated_loss: float,
        protected_value: float,
        executive_summary: str,
        adjusted_risk_score: float | None = None,
        adjusted_risk_level: str | None = None,
        final_verdict: str | None = None,
    ) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        transaction_amt = float(transaction.get("TransactionAmt", 0) or 0)

        has_adjusted = (
            adjusted_risk_score is not None
            and adjusted_risk_level is not None
            and final_verdict is not None
        )

        primary_verdict = final_verdict if has_adjusted else base_model_verdict
        primary_risk_level = adjusted_risk_level if has_adjusted else base_risk_level
        primary_score = adjusted_risk_score if has_adjusted else base_probability

        lines = [
            "=" * 60,
            "  FRAUDSENTINEL AI -- TRANSACTION VERDICT",
            "=" * 60,
            f"  Timestamp              : {timestamp}",
            f"  Model                  : {model_name}",
            "",
            f"  FINAL VERDICT          : {primary_verdict}",
            f"  FINAL RISK LEVEL       : {primary_risk_level}",
            f"  ADJUSTED RISK SCORE    : {primary_score * 100:.2f}%",
            f"  BASE MODEL PROBABILITY : {base_probability * 100:.2f}%",
            "",
            f"  Transaction Amt        : ${transaction_amt:,.2f}",
            f"  Estimated Loss         : ${estimated_loss:,.2f}",
            f"  Protected Value        : ${protected_value:,.2f}",
            "",
            "=" * 60,
            "  EXECUTIVE SUMMARY",
            "=" * 60,
            executive_summary.strip(),
            "=" * 60,
            "",
            f"  Base Model Verdict     : {base_model_verdict}",
            f"  Base Model Risk Level  : {base_risk_level}",
        ]
        return "\n".join(lines)

    def _build_adjusted_executive_summary(
        self,
        transaction_amt: float,
        base_probability: float,
        base_risk_level: str,
        base_model_verdict: str,
        adjusted_payload: dict | None = None,
    ) -> str:
        if adjusted_payload:
            final_verdict = adjusted_payload["final_verdict"]
            adjusted_risk_level = adjusted_payload["adjusted_risk_level"]
            adjusted_risk_score = adjusted_payload["adjusted_risk_score"]
            signal_count = adjusted_payload.get("signal_count", 0)

            if adjusted_risk_score > base_probability:
                escalation_text = (
                    f"The ensemble model produced a base fraud probability of {base_probability*100:.2f}%, "
                    f"and the inference-time risk layer escalated the final risk score to "
                    f"{adjusted_risk_score*100:.2f}% due to {signal_count} high-risk signal(s)."
                )
            else:
                escalation_text = (
                    f"The ensemble model produced a base fraud probability of {base_probability*100:.2f}%, "
                    f"and the final risk score remained aligned at {adjusted_risk_score*100:.2f}% after "
                    f"reviewing inference-time signals."
                )

            return (
                f"This transaction was flagged as {final_verdict} with a {adjusted_risk_level} adjusted risk level.\n"
                f"{escalation_text}"
            )

        return (
            f"This transaction was classified as {base_model_verdict} with a {base_risk_level} risk level.\n"
            f"The ensemble model produced a fraud probability of {base_probability*100:.2f}% "
            f"for a transaction amount of ${transaction_amt:,.2f}."
        )

    def predict_single(self, transaction: dict) -> dict:
        artifacts = self._load_artifacts()

        raw_df = map_basic_transaction_input(transaction)
        raw_row = raw_df.iloc[0].to_dict()

        processed_X = preprocess_for_inference(raw_df, artifacts["feature_pipeline"])
        base_prob = float(
            predict_with_stack(
                artifacts["meta_learner"],
                artifacts["xgb_model"],
                artifacts["lgb_model"],
                artifacts["cat_model"],
                processed_X,
            )[0]
        )

        adjusted = self._apply_risk_adjustment(
            raw_row, base_prob, artifacts["feature_pipeline"]
        )

        # ------------------------------
        # ORIGINAL MODEL DECISION (unchanged)
        # ------------------------------
        result = self.decision_agent.analyze_transaction(
            transaction=raw_row,
            best_model=artifacts["meta_learner"],
            feature_names=artifacts["feature_pipeline"]["feature_names"],
            model_name="Stacking Ensemble",
            xgb_model=artifacts["xgb_model"],
            lgb_model=artifacts["lgb_model"],
            cat_model=artifacts["cat_model"],
            processed_features=processed_X,
            fraud_probability=base_prob,
        )

        # ------------------------------
        # APPLY ADJUSTED LAYER
        # ------------------------------
        result["adjusted_risk_score"] = adjusted["adjusted_risk_score"]
        result["adjusted_risk_level"] = adjusted["adjusted_risk_level"]
        result["final_verdict"] = adjusted["final_verdict"]
        result["risk_adjustment"] = adjusted

        # ------------------------------
        # 🧠 NEW: FILTERED USER CONTEXT (CRITICAL FIX)
        # ------------------------------
        visible_context = filter_user_visible_context(
            raw_row,
            adjusted["applied_signals"]
        )

        result["visible_context"] = visible_context

        # ------------------------------
        # 🧠 NEW: GROUNDED EXPLANATION (OVERRIDE)
        # ------------------------------
        result["explanation"] = self.decision_agent.generate_grounded_explanation(
            visible_context=visible_context,
            base_probability=base_prob,
            adjusted_payload=adjusted
        )

        # ------------------------------
        # 🧠 FIXED COUNTERFACTUAL (NO HIDDEN FEATURES)
        # ------------------------------
        final_is_fraud = adjusted["adjusted_risk_score"] >= 0.50

        if final_is_fraud:
            cf_result = self.cf_agent.generate_counterfactual_for_transaction(
                visible_context=visible_context,
                adjusted_payload=adjusted
            )
        else:
            cf_result = None

        result["counterfactual"] = cf_result
        result["training_metadata"] = artifacts["metadata"]

        # ------------------------------
        # FINANCIALS
        # ------------------------------
        transaction_amt = float(raw_row.get("TransactionAmt", 0) or 0)
        estimated_loss = round(transaction_amt if final_is_fraud else 0.0, 2)
        protected_value = estimated_loss if final_is_fraud else round(transaction_amt, 2)

        # ------------------------------
        # BASE MODEL INTERPRETATION (DE-MOTED)
        # ------------------------------
        base_risk_level = result.get(
            "risk_level",
            self.decision_agent._risk_level_from_probability(base_prob),
        )
        base_model_verdict = result.get(
            "verdict",
            "FRAUD DETECTED" if base_prob >= 0.50 else "LEGITIMATE",
        )

        # ------------------------------
        # 🧠 UPDATED EXECUTIVE SUMMARY (USES ADJUSTED LAYER)
        # ------------------------------
        executive_summary = self._build_adjusted_executive_summary(
            transaction_amt=transaction_amt,
            base_probability=base_prob,
            base_risk_level=base_risk_level,
            base_model_verdict=base_model_verdict,
            adjusted_payload=adjusted,
        )

        result["executive_summary"] = executive_summary

        # ------------------------------
        # FINAL REPORT (ALREADY CORRECT)
        # ------------------------------
        result["report"] = self._build_single_transaction_report(
            transaction=raw_row,
            model_name=result.get("model_name", "Stacking Ensemble"),
            base_probability=base_prob,
            base_risk_level=base_risk_level,
            base_model_verdict=base_model_verdict,
            estimated_loss=estimated_loss,
            protected_value=protected_value,
            executive_summary=executive_summary,
            adjusted_risk_score=result.get("adjusted_risk_score"),
            adjusted_risk_level=result.get("adjusted_risk_level"),
            final_verdict=result.get("final_verdict"),
        )

        return result

    def predict_csv(self, csv_path: str, mode: str = "rich") -> dict:
        artifacts = self._load_artifacts()

        try:
            raw_df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Unable to read CSV file: {e}")

        if raw_df is None or raw_df.empty:
            raise ValueError("CSV input is empty. Provide at least one transaction row.")

        # --------------------------------------------------
        # Stronger CSV alias normalization
        # --------------------------------------------------
        csv_aliases = {
            "transaction_amount": "amount",
            "amt": "amount",
            "amount": "amount",
            "network": "card_network",
            "card_network": "card_network",
            "payment_card_type": "card_type",
            "card_type": "card_type",
            "billing_region": "billing_country",
            "billing_location": "billing_country",
            "billing_country": "billing_country",
            "shipping_region": "shipping_country",
            "shipping_location": "shipping_country",
            "shipping_country": "shipping_country",
            "payer_email_domain": "email_domain",
            "sender_email_domain": "email_domain",
            "buyer_email_domain": "email_domain",
            "email_domain": "email_domain",
            "recipient_email_domain": "receiver_email_domain",
            "receiver_email_domain": "receiver_email_domain",
            "merchant_email_domain": "receiver_email_domain",
            "tx_hour": "transaction_hour",
            "hour": "transaction_hour",
            "transaction_hour": "transaction_hour",
        }

        rename_map = {}
        seen_targets = set()
        for col in raw_df.columns:
            normalized_col = str(col).strip().lower()
            target = csv_aliases.get(normalized_col)
            if target and target not in raw_df.columns and target not in seen_targets:
                rename_map[col] = target
                seen_targets.add(target)

        raw_df = raw_df.rename(columns=rename_map)

        # --------------------------------------------------
        # Row-wise flexible normalization
        # --------------------------------------------------
        normalized_rows = []
        valid_indices = []
        errors = []

        for idx, row in raw_df.iterrows():
            row_dict = {}
            for k, v in row.to_dict().items():
                if pd.isna(v):
                    continue
                row_dict[k] = v

            try:
                mapped_df = map_basic_transaction_input(row_dict)
                normalized_rows.append(mapped_df.iloc[0].to_dict())
                valid_indices.append(idx)
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")

        if not normalized_rows:
            raise ValueError(
                "No valid transaction rows found in CSV. "
                "Ensure at least one row contains usable transaction fields such as amount, "
                "transaction_amount, amt, or TransactionAmt."
            )

        # --------------------------------------------------
        # Build output only from valid rows
        # --------------------------------------------------
        out = raw_df.iloc[valid_indices].copy().reset_index(drop=True)
        processed_input_df = pd.DataFrame(normalized_rows).reset_index(drop=True)

        # --------------------------------------------------
        # Model inference
        # --------------------------------------------------
        processed_X = preprocess_for_inference(
            processed_input_df,
            artifacts["feature_pipeline"]
        )

        probabilities = predict_with_stack(
            artifacts["meta_learner"],
            artifacts["xgb_model"],
            artifacts["lgb_model"],
            artifacts["cat_model"],
            processed_X,
        )

        # Safety check to prevent length mismatch crashes
        if not (
            len(out) == len(processed_input_df) == len(probabilities)
        ):
            raise RuntimeError(
                "Internal CSV normalization error: output row counts do not match "
                "normalized input and prediction lengths."
            )

        out["FraudProbability"] = [round(float(p), 6) for p in probabilities]

        # --------------------------------------------------
        # Adjusted risk layer
        # --------------------------------------------------
        adjusted_results = []

        for i, row in processed_input_df.iterrows():
            adjusted = self._apply_risk_adjustment(
                row.to_dict(),
                float(probabilities[i]),
                artifacts["feature_pipeline"],
            )
            adjusted_results.append(adjusted)

        if len(adjusted_results) != len(out):
            raise RuntimeError(
                "Internal CSV normalization error: adjusted results length does not match output rows."
            )

        out["AdjustedRiskScore"] = [round(x["adjusted_risk_score"], 4) for x in adjusted_results]
        out["AdjustedRiskLevel"] = [x["adjusted_risk_level"] for x in adjusted_results]
        out["FinalVerdict"] = [x["final_verdict"] for x in adjusted_results]

        out["RiskLevel"] = out["AdjustedRiskLevel"]
        out["Verdict"] = out["FinalVerdict"]

        out["PredictedLabel"] = (out["AdjustedRiskScore"] >= 0.50).astype(int)

        # --------------------------------------------------
        # Financial metrics
        # --------------------------------------------------
        amt_series = pd.to_numeric(
            processed_input_df.get("TransactionAmt", 0),
            errors="coerce"
        ).fillna(0)

        out["EstimatedLoss_USD"] = (
            ((out["PredictedLabel"] == 1).astype(int).reset_index(drop=True) * amt_series.reset_index(drop=True))
            .round(2)
        )
        out["RecoveredValue_USD"] = out["EstimatedLoss_USD"]

        out["FinancialImpactEstimate"] = [
            self._risk_impact_label(r, e)
            for r, e in zip(out["AdjustedRiskLevel"], out["EstimatedLoss_USD"])
        ]

        out["Recommendation"] = out["FinalVerdict"].map(
            {
                "FRAUD DETECTED": "Immediate block and analyst review",
                "HIGH RISK - MANUAL REVIEW": "Manual review required",
                "SUSPICIOUS - STEP-UP VERIFICATION": "Step-up verification required",
                "LEGITIMATE": "Allow transaction",
            }
        ).fillna("Manual review required")

        # --------------------------------------------------
        # Rich mode explanations
        # --------------------------------------------------
        if mode == "rich":
            explanations = []
            counterfactuals = []
            executive_summaries = []

            for i, row in processed_input_df.iterrows():
                raw_row = row.to_dict()
                base_prob = float(probabilities[i])
                adjusted = adjusted_results[i]

                visible_context = filter_user_visible_context(
                    raw_row,
                    adjusted["applied_signals"]
                )

                explanation = self.decision_agent.generate_grounded_explanation(
                    visible_context=visible_context,
                    base_probability=base_prob,
                    adjusted_payload=adjusted
                )

                if adjusted["adjusted_risk_score"] >= 0.50:
                    cf = self.cf_agent.generate_counterfactual_for_transaction(
                        visible_context=visible_context,
                        adjusted_payload=adjusted
                    )
                    cf_text = cf.get("llm_explanation", "")
                else:
                    cf_text = "No counterfactual needed — transaction is classified as legitimate."

                exec_summary = self._build_adjusted_executive_summary(
                    transaction_amt=float(raw_row.get("TransactionAmt", 0) or 0),
                    base_probability=base_prob,
                    base_risk_level=self.decision_agent._risk_level_from_probability(base_prob),
                    base_model_verdict="FRAUD DETECTED" if base_prob >= 0.50 else "LEGITIMATE",
                    adjusted_payload=adjusted,
                )

                explanations.append(explanation)
                counterfactuals.append(cf_text)
                executive_summaries.append(exec_summary)

            if not (
                len(explanations) == len(counterfactuals) == len(executive_summaries) == len(out)
            ):
                raise RuntimeError(
                    "Internal CSV rich-mode error: explanation output lengths do not match output rows."
                )

            out["LLM_Explanation"] = explanations
            out["CounterfactualSummary"] = counterfactuals
            out["ExecutiveSummary"] = executive_summaries

        # --------------------------------------------------
        # Save outputs
        # --------------------------------------------------
        os.makedirs(INFERENCE_OUTPUT_DIR, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            INFERENCE_OUTPUT_DIR,
            f"batch_predictions_{mode}_{ts}.csv"
        )
        out.to_csv(out_path, index=False)

        summary = self._aggregate_batch_summary(out)
        summary_path = os.path.join(
            INFERENCE_OUTPUT_DIR,
            f"batch_summary_{mode}_{ts}.json"
        )

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return {
            "rows_scored": len(out),
            "valid_rows": len(valid_indices),
            "invalid_rows": len(errors),
            "errors": errors[:10],
            "output_csv": out_path,
            "summary_json": summary_path,
            "batch_summary": summary,
            "training_metadata": artifacts["metadata"],
        }