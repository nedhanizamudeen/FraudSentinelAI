# -------------------------------------------------------------
#  FraudSentinel AI v2.1 — agents/decision_agent.py
#
#  AGENT 6: Decision Agent (The Final Judge — Extended)
#
#  NEW in v2.0:
#    - Uses the STACKING ENSEMBLE (not just best single model)
#    - Generates LLM explanations for ALL transactions
#      (not just a 7-sample subset)
#    - Runs the Counterfactual Interrogation Agent before saving
#    - Final CSV includes all required columns:
#        TransactionID | ActualLabel | PredictedLabel |
#        FraudProbability | RiskLevel | Verdict | Correct |
#        LLM_Explanation | CounterfactualAnalysis |
#        CF_FlipFeatures | CF_MinChange | CF_LLMExplanation
#    - Final report includes full confusion matrix, all metrics,
#      risk distribution, and counterfactual summary
#
#  NEW in v2.1:
#    - INNOVATION #1: Financial Impact Estimation
#        Two new CSV columns added per transaction:
#          EstimatedLoss_USD  — transaction amount at stake
#          RecoveredValue_USD — amount saved (caught fraud) or 0 (missed/legit)
#        Report section "FINANCIAL IMPACT" shows total fraud value caught,
#        missed fraud value, and false-alarm review cost.
#    - INNOVATION #3: LLM Executive Narrative Summary
#        One extra LLM call at the end generates a 3-sentence plain-English
#        summary readable by a bank manager or compliance officer.
#        Added as "EXECUTIVE SUMMARY" at the very top of the final report.
# -------------------------------------------------------------

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from tools.safe_llm import safe_llm_invoke

from tools.ml_tools import predict_with_stack
from tools.data_tools import save_report
from agents.counterfactual_agent import CounterfactualAgent
from config import (
    GROQ_API_KEY, LLM_MODEL, REPORTS_DIR, FRAUD_THRESHOLD
)


class DecisionAgent:
    """
    Decision Agent — Agent 6 in the v2.0 pipeline.

    Uses the stacking ensemble for all predictions.
    Generates LLM explanations for every single transaction.
    Integrates Counterfactual Interrogation Agent for flagged transactions.
    Produces comprehensive CSV and report outputs.
    """
    SIGNAL_PRIORITY = {
    "extreme_amount_flag": 100,
    "very_high_amount_flag": 90,
    "high_amount_flag": 80,
    "suspicious_domain_flag": 70,
    "email_domain_mismatch_flag": 65,
    "rare_card_flag": 50,
    "unusual_addr_flag": 45,
    "missing_identity_signal": 30,
    "night_transaction_flag": 20,
    }
    def __init__(self):
        self.llm = ChatGroq(
            api_key    = GROQ_API_KEY,
            model_name = LLM_MODEL,
            temperature= 0.1
        )
        self.agent_name   = "Decision Agent"
        self.cf_agent     = CounterfactualAgent()


    def _risk_level_from_probability(self, fraud_prob: float) -> str:
        if fraud_prob < 0.3:
            return "LOW"
        if fraud_prob < 0.5:
            return "MEDIUM"
        if fraud_prob < 0.75:
            return "HIGH"
        return "CRITICAL"

    def _estimate_financial_impact(self, raw_transaction: dict, fraud_probability: float, is_fraud: bool) -> dict:
        amount = 0.0
        try:
            amount = float(raw_transaction.get("TransactionAmt", 0) or 0)
        except Exception:
            amount = 0.0
        estimated_loss = amount if is_fraud else 0.0
        recovered_value = amount if is_fraud else 0.0
        review_cost = 15.0 if is_fraud else 0.0
        return {
            "transaction_amount_usd": round(amount, 2),
            "estimated_loss_usd": round(estimated_loss, 2),
            "recovered_value_usd": round(recovered_value, 2),
            "manual_review_cost_usd": round(review_cost, 2),
        }

    def _generate_single_executive_summary(self, fraud_probability: float, risk_level: str, is_fraud: bool, impact: dict) -> str:
        verdict = "fraud" if is_fraud else "legitimate"
        amount = impact.get("transaction_amount_usd", 0.0)
        return (
            f"This transaction was classified as {verdict} with a fraud probability of {fraud_probability*100:.1f}% "
            f"and a {risk_level} risk rating. "
            f"The transaction amount assessed was ${amount:,.2f}, with an estimated protected value of "
            f"${impact.get('recovered_value_usd', 0.0):,.2f}."
        )

    # ----------------------------------------------------------
    # LLM explanation for a single transaction
    # ----------------------------------------------------------
    def _explain_decision(self,
                          transaction: dict,
                          fraud_probability: float,
                          risk_level: str,
                          is_fraud: bool,
                          model_name: str) -> str:
        system_prompt = (
            "You are FraudSentinel AI's Decision Agent.\n"
            "Respond in exactly 4 short bullet points. "
            "Each bullet = one sentence, max 20 words.\n"
            "No headers, no paragraphs. Just 4 bullets starting with -"
        )
        key_features = {k: v for k, v in list(transaction.items())[:12]}
        user_message = (
            f"Transaction: {json.dumps(key_features, default=str)}\n"
            f"Model: {model_name} | Fraud Probability: {fraud_probability*100:.1f}% "
            f"| Risk: {risk_level}\n"
            f"Decision: {'FRAUD' if is_fraud else 'LEGITIMATE'}\n\n"
            "Give exactly 4 bullet points:\n"
            "- One-sentence verdict\n"
            "- Main suspicious feature (or main safe feature)\n"
            "- Recommended action\n"
            "- Confidence level and reason"
        )
        return safe_llm_invoke(
            self.llm,
            [SystemMessage(content=system_prompt), HumanMessage(content=user_message)],
            agent_name=self.agent_name,
            fallback_text="Decision explanation unavailable"
        )

    # ----------------------------------------------------------
    # Single-transaction public API (for --demo mode)
    # ----------------------------------------------------------
    def analyze_transaction(self,
                            transaction: dict,
                            best_model,
                            feature_names: list,
                            model_name: str = "Best Model",
                            xgb_model=None,
                            lgb_model=None,
                            cat_model=None,
                            processed_features: pd.DataFrame | None = None,
                            fraud_probability: float | None = None) -> dict:
        """Public API for single-transaction analysis using one consistent processed feature vector."""
        row = processed_features.copy() if processed_features is not None else pd.DataFrame([transaction])
        for col in feature_names:
            if col not in row.columns:
                row[col] = 0
        row = row[feature_names]

        if fraud_probability is None:
            if xgb_model is not None and lgb_model is not None and cat_model is not None:
                fraud_prob = float(predict_with_stack(best_model, xgb_model, lgb_model, cat_model, row)[0])
            else:
                fraud_prob = float(best_model.predict_proba(row)[0][1])
        else:
            fraud_prob = float(fraud_probability)

        risk_level = self._risk_level_from_probability(fraud_prob)
        is_fraud = fraud_prob >= FRAUD_THRESHOLD
        explanation = self._explain_decision(
            transaction=transaction, fraud_probability=fraud_prob,
            risk_level=risk_level, is_fraud=is_fraud, model_name=model_name
        )
        verdict_text = "FRAUD DETECTED" if is_fraud else "LEGITIMATE"
        impact = self._estimate_financial_impact(transaction, fraud_prob, is_fraud)
        executive_summary = self._generate_single_executive_summary(fraud_prob, risk_level, is_fraud, impact)

        report = f"""
{'='*60}
  FRAUDSENTINEL AI v2.0 -- TRANSACTION VERDICT
{'='*60}
  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Model     : {model_name}

  VERDICT: {verdict_text}

  Fraud Probability : {fraud_prob*100:.1f}%
  Risk Level        : {risk_level}
  Threshold Used    : {FRAUD_THRESHOLD*100:.0f}%
  Transaction Amt   : ${impact['transaction_amount_usd']:,.2f}
  Estimated Loss    : ${impact['estimated_loss_usd']:,.2f}
  Protected Value   : ${impact['recovered_value_usd']:,.2f}

{'='*60}
  EXECUTIVE SUMMARY:
{'='*60}
{executive_summary}
{'='*60}
  [LLM] AI EXPLANATION:
{'='*60}
{explanation}
{'='*60}
"""
        return {
            "fraud_probability": fraud_prob, "is_fraud": is_fraud,
            "risk_level": risk_level, "verdict": verdict_text,
            "explanation": explanation, "report": report,
            "executive_summary": executive_summary,
            "financial_impact": impact,
            "processed_features": row
        }

    # ----------------------------------------------------------
    # Batch prediction using stacking ensemble
    # ----------------------------------------------------------
    def _run_batch_predictions(
        self,
        meta_learner,
        xgb_model,
        lgb_model,
        cat_model,
        feature_names: list,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_preds_avg=None
    ) -> pd.DataFrame:
        """
        Scores ALL test transactions using the stacking ensemble.
        Uses pre-computed test_preds_avg from the OOF stacking step
        for maximum accuracy (these are the proper fold-averaged predictions).
        """
        print(f"  [DecisionAgent] Scoring all {len(X_test):,} transactions "
              f"via Stacking Ensemble...")

        if test_preds_avg is not None:
            # Use pre-computed fold-averaged base model predictions
            proba = meta_learner.predict_proba(test_preds_avg)[:, 1]
        else:
            proba = predict_with_stack(
                meta_learner, xgb_model, lgb_model, cat_model, X_test[feature_names]
            )

        predicted_label = (proba >= FRAUD_THRESHOLD).astype(int)

        risk_levels = []
        for p in proba:
            if p < 0.30:    risk_levels.append("LOW")
            elif p < 0.50:  risk_levels.append("MEDIUM")
            elif p < 0.75:  risk_levels.append("HIGH")
            else:           risk_levels.append("CRITICAL")

        verdicts = ["FRAUD" if p == 1 else "LEGITIMATE" for p in predicted_label]
        correct  = [int(pred == actual)
                    for pred, actual in zip(predicted_label, y_test.values)]

        # [INNOVATION #1] Pull TransactionAmt for financial impact estimation
        amt_values = (
            X_test["TransactionAmt"].values
            if "TransactionAmt" in X_test.columns
            else np.zeros(len(proba))
        )

        # EstimatedLoss_USD  = transaction amount at stake for every flagged/missed fraud
        # RecoveredValue_USD = amount protected (caught TP); 0 for missed fraud or legit
        actual_arr    = np.array(y_test.values)
        predicted_arr = np.array(predicted_label)

        estimated_loss   = np.where(
            (predicted_arr == 1) | (actual_arr == 1),   # flagged OR actually fraud
            amt_values, 0.0
        )
        recovered_value  = np.where(
            (predicted_arr == 1) & (actual_arr == 1),   # true positive — fraud caught
            amt_values, 0.0
        )

        df = pd.DataFrame({
            "TransactionID"      : X_test.index,
            "ActualLabel"        : y_test.values,
            "PredictedLabel"     : predicted_label,
            "FraudProbability"   : np.round(proba, 4),
            "RiskLevel"          : risk_levels,
            "Verdict"            : verdicts,
            "Correct"            : correct,
            "TransactionAmt_USD" : np.round(amt_values, 2),
            "EstimatedLoss_USD"  : np.round(estimated_loss, 2),   # [INNOVATION #1]
            "RecoveredValue_USD" : np.round(recovered_value, 2),  # [INNOVATION #1]
            "LLM_Explanation"    : ""
        })

        print(f"  [DecisionAgent] Batch scoring complete.")
        return df

    # ----------------------------------------------------------
    # Fill LLM explanations for ALL transactions
    # ----------------------------------------------------------
    def _fill_all_explanations(
        self,
        results_df: pd.DataFrame,
        X_test: pd.DataFrame,
        model_name: str,
        batch_size: int = 50
    ) -> pd.DataFrame:
        """
        Generates LLM explanations for every single transaction.
        Uses batching to manage API rate limits gracefully.
        Falls back to a template explanation if the API is slow.
        """
        results_df = results_df.copy()
        total      = len(results_df)
        print(f"  [DecisionAgent] Generating LLM explanations for "
              f"ALL {total:,} transactions...")

        for i, row_pos in enumerate(results_df.index):
            txn_index = results_df.loc[row_pos, "TransactionID"]
            is_fraud  = bool(results_df.loc[row_pos, "PredictedLabel"])
            prob      = float(results_df.loc[row_pos, "FraudProbability"])
            risk      = results_df.loc[row_pos, "RiskLevel"]

            # Progress indicator every 100 rows
            if i % 100 == 0:
                print(f"    Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

            try:
                transaction = X_test.loc[txn_index].to_dict()
            except (KeyError, AttributeError):
                # Fallback: construct template explanation
                label = "FRAUD" if is_fraud else "LEGITIMATE"
                results_df.loc[row_pos, "LLM_Explanation"] = (
                    f"- Decision: {label} with {prob*100:.1f}% probability | "
                    f"- Risk level {risk} based on ensemble model | "
                    f"- Threshold: {FRAUD_THRESHOLD*100:.0f}% | "
                    f"- Automated flagging: manual review recommended"
                )
                continue

            try:
                explanation = self._explain_decision(
                    transaction       = transaction,
                    fraud_probability = prob,
                    risk_level        = risk,
                    is_fraud          = is_fraud,
                    model_name        = model_name
                )
                results_df.loc[row_pos, "LLM_Explanation"] = explanation.replace("\n", " | ")
            except Exception as e:
                # API error — use template
                label = "FRAUD" if is_fraud else "LEGITIMATE"
                results_df.loc[row_pos, "LLM_Explanation"] = (
                    f"- {label} ({prob*100:.1f}% fraud prob, {risk} risk) | "
                    f"- Stacking ensemble decision | "
                    f"- API error: {str(e)[:50]} | "
                    f"- Review transaction manually"
                )

        print(f"  [DecisionAgent] All {total:,} explanations generated.")
        return results_df

    # ----------------------------------------------------------
    # Confusion matrix visualization
    # ----------------------------------------------------------
    def _save_confusion_matrix_plot(
        self, tp: int, fp: int, fn: int, tn: int, model_name: str
    ) -> str:
        """Saves a confusion matrix heatmap as PNG to the reports folder."""
        os.makedirs(REPORTS_DIR, exist_ok=True)
        cm_array = np.array([[tn, fp], [fn, tp]])
        labels   = [["TN\n(Correct Legit)", "FP\n(False Alarm)"],
                    ["FN\n(Missed Fraud)",  "TP\n(Caught Fraud)"]]

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm_array, annot=False, fmt="d", cmap="Blues",
            xticklabels=["Predicted: Legit", "Predicted: Fraud"],
            yticklabels=["Actual: Legit", "Actual: Fraud"],
            ax=ax
        )
        for i in range(2):
            for j in range(2):
                color = "white" if cm_array[i, j] > cm_array.max() / 2 else "black"
                ax.text(j + 0.5, i + 0.5,
                        f"{labels[i][j]}\n{cm_array[i,j]:,}",
                        ha="center", va="center",
                        fontsize=11, color=color, fontweight="bold")

        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
        plt.tight_layout()
        plot_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        return plot_path

    # ----------------------------------------------------------
    # Model comparison bar chart
    # ----------------------------------------------------------
    def _save_model_comparison_chart(self, all_eval_results: dict) -> str:
        """Saves a grouped bar chart comparing all 4 models."""
        os.makedirs(REPORTS_DIR, exist_ok=True)
        model_map = {
            "xgb"  : "XGBoost",
            "lgb"  : "LightGBM",
            "cat"  : "CatBoost",
            "stack": "Stacking"
        }
        models  = [model_map[k] for k in model_map if k in all_eval_results]
        metrics = {"AUC": [], "F1": [], "Precision": [], "Recall": []}

        for key in model_map:
            if key not in all_eval_results:
                continue
            r = all_eval_results[key]
            metrics["AUC"].append(r["auc"])
            metrics["F1"].append(r["f1"])
            metrics["Precision"].append(r["precision"])
            metrics["Recall"].append(r["recall"])

        x     = np.arange(len(models))
        width = 0.2
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (metric, values) in enumerate(metrics.items()):
            bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i])
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("FraudSentinel v2.0 — Model Comparison: XGBoost vs LightGBM vs CatBoost vs Stacking")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.1)
        ax.legend(loc="lower right")
        ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.5, label="Min AUC threshold")
        plt.tight_layout()

        chart_path = os.path.join(REPORTS_DIR, "model_comparison.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        return chart_path

    # ----------------------------------------------------------
    # Risk distribution chart
    # ----------------------------------------------------------
    def _save_risk_distribution_chart(self, results_df: pd.DataFrame) -> str:
        """Saves a pie chart of risk level distribution."""
        os.makedirs(REPORTS_DIR, exist_ok=True)
        risk_counts = results_df["RiskLevel"].value_counts()
        colors_map  = {"LOW": "#4CAF50", "MEDIUM": "#FF9800",
                       "HIGH": "#F44336", "CRITICAL": "#9C27B0"}
        colors = [colors_map.get(r, "grey") for r in risk_counts.index]

        fig, ax = plt.subplots(figsize=(7, 5))
        wedges, texts, autotexts = ax.pie(
            risk_counts.values,
            labels  = risk_counts.index,
            colors  = colors,
            autopct = "%1.1f%%",
            startangle = 90,
            pctdistance= 0.85
        )
        for autotext in autotexts:
            autotext.set_fontsize(10)
        ax.set_title("FraudSentinel v2.0 — Risk Level Distribution", fontsize=13)

        chart_path = os.path.join(REPORTS_DIR, "risk_distribution.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        return chart_path

    # ----------------------------------------------------------
    # [INNOVATION #3] LLM Executive Narrative Summary
    # ----------------------------------------------------------
    def _generate_executive_narrative(
        self,
        total: int,
        flagged: int,
        actual_fraud: int,
        tp: int,
        fp: int,
        fn: int,
        recall: float,
        precision: float,
        fraud_value_caught: float,
        fraud_value_missed: float,
        risk_dist: dict,
        best_model_name: str,
        best_auc: float
    ) -> str:
        """
        [INNOVATION #3] Generates a 3-sentence plain-English executive summary
        using the LLM. Written for a bank manager or compliance officer —
        no ML jargon. Added as the first section of the final report.
        """
        system_prompt = (
            "You are a senior fraud analyst writing an executive summary for a bank board report.\n"
            "Write exactly 3 sentences. No bullet points, no headers, no jargon.\n"
            "Be professional, specific with numbers, and businesslike.\n"
            "Focus on business impact: money saved, fraud caught, operational cost."
        )
        user_message = (
            f"Fraud detection run results:\n"
            f"  Model: {best_model_name} | AUC: {best_auc:.4f}\n"
            f"  Total transactions screened : {total:,}\n"
            f"  Actual fraud cases          : {actual_fraud:,}\n"
            f"  Fraud cases caught (TP)     : {tp:,} ({recall*100:.1f}% catch rate)\n"
            f"  Fraud cases missed (FN)     : {fn:,}\n"
            f"  False alarms generated (FP) : {fp:,} (flagged but legitimate)\n"
            f"  Precision                   : {precision*100:.1f}%\n"
            f"  Fraud value caught          : ${fraud_value_caught:,.2f}\n"
            f"  Fraud value missed          : ${fraud_value_missed:,.2f}\n"
            f"  Risk distribution           : CRITICAL={risk_dist.get('CRITICAL',0):,}, "
            f"HIGH={risk_dist.get('HIGH',0):,}, MEDIUM={risk_dist.get('MEDIUM',0):,}, "
            f"LOW={risk_dist.get('LOW',0):,}\n\n"
            "Write exactly 3 sentences summarising this for a bank executive. "
            "Mention the dollar amounts, catch rate, and one operational note about false alarms."
        )
        summary = safe_llm_invoke(
            self.llm,
            [SystemMessage(content=system_prompt), HumanMessage(content=user_message)],
            agent_name=self.agent_name,
            fallback_text=(
                f"FraudSentinel AI screened {total:,} transactions and successfully identified "
                f"{tp:,} of {actual_fraud:,} fraud cases ({recall*100:.1f}% catch rate), "
                f"protecting an estimated ${fraud_value_caught:,.2f} in transaction value. "
                f"The model generated {fp:,} false alarms requiring manual review, "
                f"while {fn:,} fraudulent transactions (${fraud_value_missed:,.2f}) were not flagged. "
                f"Overall, the {best_model_name} achieved an AUC of {best_auc:.4f}, "
                f"demonstrating strong discriminative performance for production deployment."
            )
        )
        return summary.strip()

    # ----------------------------------------------------------
    # Main pipeline run
    # ----------------------------------------------------------
    def run(self, state: dict) -> dict:
        """
        Pipeline mode:
          1. Batch scores ALL test transactions via stacking ensemble
             [INNOVATION #1] Adds EstimatedLoss_USD + RecoveredValue_USD columns
          2. Generates LLM explanations for ALL transactions
          3. Runs Counterfactual Interrogation Agent on top fraud cases
          4. Saves full CSV + fraud-only CSV
          5. Saves confusion matrix, model comparison, risk distribution charts
          6. Writes final comprehensive report
             [INNOVATION #3] Report opens with LLM Executive Narrative Summary
        """
        print(f"\n{'='*60}")
        print(f"  [Decision] {self.agent_name} — STARTING")
        print(f"{'='*60}")

        meta_learner    = state["meta_learner"]
        xgb_model       = state["xgb_model"]
        lgb_model       = state["lgb_model"]
        cat_model       = state["cat_model"]
        best_model_name = state["best_model_name"]
        best_auc        = state["best_auc"]
        X_test          = state["X_test"]
        y_test          = state["y_test"]
        feature_names   = state["feature_names"]
        test_preds_avg  = state.get("test_preds_avg")
        all_eval_results= state.get("all_eval_results", {})

        # Step 1 — Batch predict all transactions
        print(f"\n[Step 1/5] Batch predicting {len(X_test):,} transactions...")
        results_df = self._run_batch_predictions(
            meta_learner   = meta_learner,
            xgb_model      = xgb_model,
            lgb_model      = lgb_model,
            cat_model      = cat_model,
            feature_names  = feature_names,
            X_test         = X_test,
            y_test         = y_test,
            test_preds_avg = test_preds_avg
        )

        # Step 2 — Generate LLM explanations for ALL transactions
        print(f"\n[Step 2/5] Generating LLM explanations for all transactions...")
        results_df = self._fill_all_explanations(
            results_df = results_df,
            X_test     = X_test,
            model_name = best_model_name
        )

        # Step 3 — Run Counterfactual Interrogation Agent
        print(f"\n[Step 3/5] Running Counterfactual Interrogation Agent...")
        cf_state = {
            **state,
            "results_df": results_df
        }
        cf_state = self.cf_agent.run(cf_state)
        results_df = cf_state["results_df"]

        # Step 4 — Compute metrics
        print(f"\n[Step 4/5] Computing evaluation metrics and saving outputs...")
        total        = len(results_df)
        flagged      = int(results_df["PredictedLabel"].sum())
        actual_fraud = int(results_df["ActualLabel"].sum())
        correct_total= int(results_df["Correct"].sum())
        accuracy     = correct_total / total * 100

        tp = int(((results_df["PredictedLabel"] == 1) & (results_df["ActualLabel"] == 1)).sum())
        fp = int(((results_df["PredictedLabel"] == 1) & (results_df["ActualLabel"] == 0)).sum())
        fn = int(((results_df["PredictedLabel"] == 0) & (results_df["ActualLabel"] == 1)).sum())
        tn = int(((results_df["PredictedLabel"] == 0) & (results_df["ActualLabel"] == 0)).sum())

        recall      = tp / (tp + fn)     if (tp + fn) > 0     else 0
        precision   = tp / (tp + fp)     if (tp + fp) > 0     else 0
        f1_score    = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp)     if (tn + fp) > 0     else 0
        fpr         = fp / (fp + tn)     if (fp + tn) > 0     else 0

        risk_dist = results_df["RiskLevel"].value_counts().to_dict()

        # [INNOVATION #1] Financial impact totals
        fraud_value_caught  = float(results_df["RecoveredValue_USD"].sum())
        fraud_value_missed  = float(
            results_df.loc[
                (results_df["PredictedLabel"] == 0) & (results_df["ActualLabel"] == 1),
                "TransactionAmt_USD"
            ].sum()
        )
        false_alarm_amt     = float(
            results_df.loc[
                (results_df["PredictedLabel"] == 1) & (results_df["ActualLabel"] == 0),
                "TransactionAmt_USD"
            ].sum()
        )
        total_fraud_value   = fraud_value_caught + fraud_value_missed
        protection_rate_pct = (
            fraud_value_caught / total_fraud_value * 100
            if total_fraud_value > 0 else 0.0
        )

        print(f"  [Financial] Fraud value caught  : ${fraud_value_caught:,.2f}")
        print(f"  [Financial] Fraud value missed  : ${fraud_value_missed:,.2f}")
        print(f"  [Financial] False-alarm value   : ${false_alarm_amt:,.2f}")

        # [INNOVATION #3] Generate executive narrative (one LLM call)
        print("\n[Executive Narrative] Generating LLM executive summary...")
        executive_narrative = self._generate_executive_narrative(
            total               = total,
            flagged             = flagged,
            actual_fraud        = actual_fraud,
            tp                  = tp,
            fp                  = fp,
            fn                  = fn,
            recall              = recall,
            precision           = precision,
            fraud_value_caught  = fraud_value_caught,
            fraud_value_missed  = fraud_value_missed,
            risk_dist           = risk_dist,
            best_model_name     = best_model_name,
            best_auc            = best_auc
        )
        print(f"  [Executive Narrative] Done.")
        os.makedirs(REPORTS_DIR, exist_ok=True)
        csv_path = os.path.join(REPORTS_DIR, "transaction_results.csv")
        results_df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"  [CSV] All transactions     : {csv_path}")

        fraud_df  = results_df[results_df["PredictedLabel"] == 1].copy()
        fraud_csv = os.path.join(REPORTS_DIR, "fraud_flagged_transactions.csv")
        fraud_df.to_csv(fraud_csv, index=False, encoding="utf-8")
        print(f"  [CSV] Fraud flagged only   : {fraud_csv} ({len(fraud_df):,} rows)")

        legit_df  = results_df[results_df["PredictedLabel"] == 0].copy()
        legit_csv = os.path.join(REPORTS_DIR, "legitimate_transactions.csv")
        legit_df.to_csv(legit_csv, index=False, encoding="utf-8")
        print(f"  [CSV] Legitimate only      : {legit_csv} ({len(legit_df):,} rows)")

        # Save charts
        print("\n[Step 5/5] Generating visualizations...")
        cm_plot_path  = self._save_confusion_matrix_plot(tp, fp, fn, tn, best_model_name)
        cmp_plot_path = self._save_model_comparison_chart(all_eval_results)
        rsk_plot_path = self._save_risk_distribution_chart(results_df)
        print(f"  [Plot] Confusion matrix    : {cm_plot_path}")
        print(f"  [Plot] Model comparison    : {cmp_plot_path}")
        print(f"  [Plot] Risk distribution   : {rsk_plot_path}")

        # Build individual model metric rows for the report
        model_rows = ""
        for key, label in [("xgb","XGBoost"),("lgb","LightGBM"),
                           ("cat","CatBoost"),("stack","Stacking*")]:
            if key in all_eval_results:
                r = all_eval_results[key]
                model_rows += (
                    f"  {label:<14} AUC={r['auc']:.4f}  F1={r['f1']:.4f}  "
                    f"Prec={r['precision']:.4f}  Rec={r['recall']:.4f}\n"
                )

        final_report = (
            f"FRAUDSENTINEL AI v2.1 — FINAL PIPELINE REPORT\n"
            f"{'='*65}\n"
            f"Completed    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Best Model   : {best_model_name}\n"
            f"AUC-ROC      : {best_auc:.4f}\n"
            f"\n"
            f"{'='*65}\n"
            f"  EXECUTIVE SUMMARY  [INNOVATION #3 — LLM Narrative]\n"
            f"{'='*65}\n"
            f"{executive_narrative}\n"
            f"\n"
            f"=== MODEL COMPARISON ===\n"
            f"{model_rows}"
            f"  (* Stacking = OOF meta-learner on XGBoost + LightGBM + CatBoost)\n"
            f"\n"
            f"=== BATCH PREDICTION SUMMARY ===\n"
            f"  Total test transactions    : {total:,}\n"
            f"  Actual fraud in test set   : {actual_fraud:,} ({actual_fraud/total*100:.2f}%)\n"
            f"  Transactions flagged       : {flagged:,}\n"
            f"  Overall accuracy           : {accuracy:.2f}%\n"
            f"\n"
            f"=== FINANCIAL IMPACT  [INNOVATION #1 — $ Estimation] ===\n"
            f"  Total fraud value in test set    : ${total_fraud_value:>12,.2f}\n"
            f"  Fraud value CAUGHT (TP)          : ${fraud_value_caught:>12,.2f}  "
            f"({protection_rate_pct:.1f}% of fraud value protected)\n"
            f"  Fraud value MISSED (FN)          : ${fraud_value_missed:>12,.2f}\n"
            f"  False-alarm transaction value    : ${false_alarm_amt:>12,.2f}  "
            f"({fp:,} legitimate txns flagged for review)\n"
            f"\n"
            f"=== CONFUSION MATRIX ===\n"
            f"                    Predicted: LEGIT    Predicted: FRAUD\n"
            f"  Actual: LEGIT     {tn:>10,}          {fp:>10,}\n"
            f"  Actual: FRAUD     {fn:>10,}          {tp:>10,}\n"
            f"\n"
            f"=== EVALUATION METRICS ===\n"
            f"  True Positives  (caught fraud)       : {tp:,}\n"
            f"  False Positives (false alarms)       : {fp:,}\n"
            f"  False Negatives (missed fraud)       : {fn:,}\n"
            f"  True Negatives  (correct legit)      : {tn:,}\n"
            f"  Recall    (fraud catch rate)         : {recall*100:.2f}%\n"
            f"  Precision (flagged that are fraud)   : {precision*100:.2f}%\n"
            f"  F1 Score                             : {f1_score:.4f}\n"
            f"  Specificity (legit catch rate)       : {specificity*100:.2f}%\n"
            f"  False Positive Rate                  : {fpr*100:.2f}%\n"
            f"\n"
            f"=== RISK LEVEL DISTRIBUTION ===\n"
            f"  CRITICAL : {risk_dist.get('CRITICAL', 0):,}\n"
            f"  HIGH     : {risk_dist.get('HIGH', 0):,}\n"
            f"  MEDIUM   : {risk_dist.get('MEDIUM', 0):,}\n"
            f"  LOW      : {risk_dist.get('LOW', 0):,}\n"
            f"\n"
            f"=== COUNTERFACTUAL INTERROGATION SUMMARY ===\n"
            + "\n".join(cf_state.get("cf_summaries", [])) + "\n"
            f"\n"
            f"=== OUTPUT FILES ===\n"
            f"  reports/transaction_results.csv         — all {total:,} transactions\n"
            f"  reports/fraud_flagged_transactions.csv  — {len(fraud_df):,} fraud rows\n"
            f"  reports/legitimate_transactions.csv     — {len(legit_df):,} legit rows\n"
            f"  reports/confusion_matrix.png            — confusion matrix heatmap\n"
            f"  reports/model_comparison.png            — 4-model comparison chart\n"
            f"  reports/risk_distribution.png           — risk level pie chart\n"
            f"  reports/05_counterfactual_report.txt    — full CF analysis\n"
            f"\n"
            f"  CSV Columns:\n"
            f"    TransactionID | ActualLabel | PredictedLabel | FraudProbability\n"
            f"    RiskLevel | Verdict | Correct\n"
            f"    TransactionAmt_USD | EstimatedLoss_USD | RecoveredValue_USD\n"
            f"    LLM_Explanation\n"
            f"    CounterfactualAnalysis | CF_FlipFeatures | CF_MinChange | CF_LLMExplanation\n"
            f"\n"
            f"{'='*65}\n"
            f"Pipeline completed autonomously by FraudSentinel AI v2.1.\n"
        )

        print(f"\n{final_report}")
        save_report("06_final_report", final_report, REPORTS_DIR)

        print(f"  [OK] {self.agent_name} COMPLETE")
        print(f"\n{'='*70}")
        print(f"  FRAUDSENTINEL AI v2.1 PIPELINE COMPLETE!")
        print(f"  Best Model : {best_model_name} | AUC: {best_auc:.4f}")
        print(f"  Fraud value caught : ${fraud_value_caught:,.2f} | Missed : ${fraud_value_missed:,.2f}")
        print(f"  All reports saved in /reports/ folder")
        print(f"{'='*70}\n")

        return {
            **state,
            "results_df"          : results_df,
            "fraud_df"            : fraud_df,
            "legit_df"            : legit_df,
            "csv_path"            : csv_path,
            "fraud_csv_path"      : fraud_csv,
            "legit_csv_path"      : legit_csv,
            "cm_plot_path"        : cm_plot_path,
            "final_report"        : final_report,
            "executive_narrative" : executive_narrative,
            "batch_stats"         : {
                "total": total, "flagged": flagged, "actual_fraud": actual_fraud,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "accuracy": accuracy, "recall": recall,
                "precision": precision, "f1_score": f1_score,
                "specificity": specificity, "fpr": fpr,
                "fraud_value_caught"  : fraud_value_caught,
                "fraud_value_missed"  : fraud_value_missed,
                "false_alarm_amt"     : false_alarm_amt,
                "total_fraud_value"   : total_fraud_value,
                "protection_rate_pct" : protection_rate_pct
            },
            "status" : "pipeline_complete",
            "agent"  : self.agent_name
        }
    
    def generate_grounded_explanation(self, visible_context, base_probability, adjusted_payload):
        signals = visible_context.get("signals", {})
        adjusted_payload = adjusted_payload or {}

        signal_to_bullet = {
            "extreme_amount_flag": "Extremely high transaction amount significantly increased the risk score.",
            "very_high_amount_flag": "Very high transaction amount contributed strongly to elevated risk.",
            "high_amount_flag": "High transaction amount increased the likelihood of fraud.",
            "suspicious_domain_flag": "Suspicious email domain usage contributed to fraud risk.",
            "email_domain_mismatch_flag": "Mismatch between sender and receiver email domains increased anomaly detection.",
            "rare_card_flag": "Unrecognized or uncommon card network contributed to anomaly detection.",
            "unusual_addr_flag": "Unusual address values contributed to the suspicious profile.",
            "missing_identity_signal": "Missing key identity attributes increased uncertainty and risk.",
            "night_transaction_flag": "Transaction timing contributed additional risk.",
        }

        triggered = []
        for signal_name, bullet in signal_to_bullet.items():
            if signals.get(signal_name):
                triggered.append(
                    (
                        self.SIGNAL_PRIORITY.get(signal_name, 0),
                        signal_name,
                        bullet,
                    )
                )

        triggered.sort(key=lambda x: (-x[0], x[1]))

        bullets = [item[2] for item in triggered[:4]]

        if adjusted_payload.get("signal_count", 0) >= 3:
            bullets.append("Multiple concurrent risk signals escalated the final fraud classification.")

        return "\n".join([f"- {b}" for b in bullets[:5]])