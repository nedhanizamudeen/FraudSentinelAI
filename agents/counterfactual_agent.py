# -------------------------------------------------------------
#  FraudSentinel AI v2.0 — agents/counterfactual_agent.py
#
#  AGENT 5 (NEW): Counterfactual Interrogation Agent
#
#  WHAT DOES THIS AGENT DO?
#  ========================
#  For each flagged (fraud) transaction, this agent asks:
#    "What would this transaction need to look like to be
#     classified as LEGITIMATE?"
#
#  TECHNICAL APPROACH: Greedy Per-Feature Perturbation
#  ====================================================
#  For each candidate feature, N_COUNTERFACTUAL_STEPS values are
#  generated across the 1st-99th percentile range from training data
#  (not raw min/max — avoids implausible extrapolation), sorted by
#  proximity to the original value so the SMALLEST change is found first.
#  The first value producing prob < FRAUD_THRESHOLD is the minimum CF.
#
#  BUGS FIXED vs INITIAL VERSION:
#  - Column name: "TransactionIndex" → "TransactionID" (matches DecisionAgent)
#  - X_train numpy array guard: _compute_feature_ranges safely handles
#    non-DataFrame inputs
#  - row construction uses only known feature_names — no KeyError on missing cols
#  - test_values sorted by proximity to original → smallest change found first
#  - LLM fallback on API error — no crash, returns structured template
#  - cf_summaries always initialised before loop — no UnboundLocalError
#  - flip_features sorted by absolute change magnitude (smallest first)
# -------------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from tools.safe_llm import safe_llm_invoke

from tools.ml_tools import predict_with_stack
from tools.data_tools import save_report
from config import (
    GROQ_API_KEY, LLM_MODEL, REPORTS_DIR,
    FRAUD_THRESHOLD, N_COUNTERFACTUAL_STEPS,
    N_COUNTERFACTUAL_TRANSACTIONS
)


class CounterfactualAgent:
    """
    Counterfactual Interrogation Agent — Agent 5 in the v2.0 pipeline.

    For each of the top N fraud-flagged transactions it:
      1. Computes realistic perturbation ranges from training data
      2. Perturbs each candidate feature one-at-a-time, sorted by
         proximity to original (smallest change tested first)
      3. Re-evaluates via stacking ensemble after each perturbation
      4. Records the smallest change that flips prediction to LEGITIMATE
      5. Generates an LLM narrative explaining the findings
      6. Writes 4 new columns back into results_df
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
        self.agent_name = "Counterfactual Interrogation Agent"

        # Candidate features ordered by typical fraud importance.
        # Only features present in the actual dataset are used.
        self.perturb_features = [
            "TransactionAmt",
            "TransactionAmt_log",
            "TransactionAmt_to_card_mean",
            "transaction_hour",
            "is_night_transaction",
            "card1_count",
            "amt_addr_ratio",
            "dist1",
            "TransactionDT",
        ]

    # ──────────────────────────────────────────────────────────
    # STEP 1: Compute perturbation ranges from training data
    # ──────────────────────────────────────────────────────────
    def _compute_feature_ranges(self, X_train) -> dict:
        """
        Computes (1st pct, 99th pct) for every candidate feature.
        Always works with a pd.DataFrame; returns {} if unavailable.
        """
        if not isinstance(X_train, pd.DataFrame):
            return {}

        ranges = {}
        for feat in self.perturb_features:
            if feat in X_train.columns:
                col = X_train[feat].dropna()
                if len(col) == 0:
                    continue
                ranges[feat] = {
                    "min": float(col.quantile(0.01)),
                    "max": float(col.quantile(0.99)),
                }
        return ranges

    # ──────────────────────────────────────────────────────────
    # STEP 2a: Perturb one feature for one transaction
    # ──────────────────────────────────────────────────────────
    def _perturb_feature(
        self,
        transaction: dict,
        feature: str,
        feat_range: dict,
        feature_names: list,
        meta_learner,
        xgb_model,
        lgb_model,
        cat_model,
    ) -> dict:
        """
        Scans N_COUNTERFACTUAL_STEPS values for `feature`, sorted by
        proximity to the original so the smallest change is found first.
        Returns a structured dict describing every step and the flip (if any).
        """
        original_val = transaction.get(feature)
        if original_val is None:
            return {
                "feature"         : feature,
                "original_value"  : None,
                "flip_found"      : False,
                "flip_value"      : None,
                "flip_probability": None,
                "change_direction": None,
                "steps_tested"    : [],
            }

        orig_f   = float(original_val)
        feat_min = feat_range["min"]
        feat_max = feat_range["max"]

        # Generate candidate values and sort by proximity to original
        raw_candidates = np.linspace(feat_min, feat_max, N_COUNTERFACTUAL_STEPS)
        candidates     = sorted(raw_candidates, key=lambda v: abs(v - orig_f))

        steps_tested = []
        flip_found   = False
        flip_value   = None
        flip_prob    = None
        change_dir   = None

        for val in candidates:
            if abs(val - orig_f) < 1e-6:
                continue   # same as original — skip

            # Build model-ready row using only known feature columns
            row_data = {col: transaction.get(col, 0) for col in feature_names}
            row_data[feature] = float(val)   # apply the perturbation
            row = pd.DataFrame([row_data])[feature_names]

            try:
                prob = float(
                    predict_with_stack(
                        meta_learner, xgb_model, lgb_model, cat_model, row
                    )[0]
                )
            except Exception:
                continue   # prediction failed for this value — skip silently

            steps_tested.append({
                "value"      : round(float(val), 4),
                "probability": round(prob, 4),
            })

            if prob < FRAUD_THRESHOLD and not flip_found:
                flip_found = True
                flip_value = round(float(val), 4)
                flip_prob  = round(prob, 4)
                change_dir = "decrease" if float(val) < orig_f else "increase"
                break   # smallest flip found — stop scanning this feature

        return {
            "feature"         : feature,
            "original_value"  : round(orig_f, 4),
            "flip_found"      : flip_found,
            "flip_value"      : flip_value,
            "flip_probability": flip_prob,
            "change_direction": change_dir,
            "steps_tested"    : steps_tested,
        }

    # ──────────────────────────────────────────────────────────
    # STEP 2b: Full CF sweep for one transaction
    # ──────────────────────────────────────────────────────────
    def _analyze_one_transaction(
        self,
        txn_index,
        transaction: dict,
        orig_prob: float,
        feature_names: list,
        feature_ranges: dict,
        meta_learner,
        xgb_model,
        lgb_model,
        cat_model,
    ) -> dict:
        """
        Runs _perturb_feature for every candidate feature with a computed
        range, collects all results, sorts flips by smallest change first,
        and calls the LLM for a natural-language explanation.
        """
        txn_clean = {k: v for k, v in transaction.items() if k != "_orig_prob"}

        perturbation_results = []
        flip_features        = []

        for feat in self.perturb_features:
            if feat not in feature_ranges:
                continue

            result = self._perturb_feature(
                transaction   = txn_clean,
                feature       = feat,
                feat_range    = feature_ranges[feat],
                feature_names = feature_names,
                meta_learner  = meta_learner,
                xgb_model     = xgb_model,
                lgb_model     = lgb_model,
                cat_model     = cat_model,
            )
            perturbation_results.append(result)
            if result["flip_found"]:
                flip_features.append(result)

        # Sort flip features: smallest absolute change magnitude first
        flip_features.sort(
            key=lambda r: (
                abs(r["flip_value"] - r["original_value"])
                if r["flip_value"] is not None and r["original_value"] is not None
                else float("inf")
            )
        )

        explanation = self._explain_counterfactual(
            transaction  = txn_clean,
            orig_prob    = orig_prob,
            flip_features= flip_features,
            all_results  = perturbation_results,
        )

        return {
            "txn_index"           : txn_index,
            "original_probability": round(orig_prob, 4),
            "flip_features"       : flip_features,
            "all_perturbations"   : perturbation_results,
            "n_flips_found"       : len(flip_features),
            "llm_explanation"     : explanation,
        }

    # ──────────────────────────────────────────────────────────
    # STEP 3: LLM narrates the counterfactual findings
    # ──────────────────────────────────────────────────────────
    def _explain_counterfactual(
        self,
        transaction: dict,
        orig_prob: float,
        flip_features: list,
        all_results: list,
    ) -> str:
        """
        Calls LLaMA 3.3 to produce a 5-bullet human-readable explanation.
        Falls back to a structured template on any API error.
        """
        system_prompt = (
            "You are the Counterfactual Interrogation Agent in FraudSentinel AI v2.0.\n"
            "Explain why a transaction was flagged and what minimal changes "
            "would make it appear legitimate.\n"
            "Respond in exactly 5 bullet points. "
            "Each bullet = one sentence, max 25 words.\n"
            "No headers, no paragraphs. Just 5 bullets starting with -"
        )

        key_txn = {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in list(transaction.items())[:10]
        }

        flip_lines = []
        for f in flip_features[:3]:
            flip_lines.append(
                f"  {f['feature']}: {f['original_value']} → {f['flip_value']} "
                f"({f['change_direction']}, new prob={f['flip_probability']})"
            )

        user_message = (
            f"Transaction (key features): {json.dumps(key_txn, default=str)}\n"
            f"Fraud probability: {orig_prob * 100:.1f}% — classified as FRAUD\n\n"
            f"Features whose perturbation flips prediction to LEGITIMATE:\n"
            + ("\n".join(flip_lines) if flip_lines else "  None found within tested range")
            + "\n\n"
            "Give exactly 5 bullet points:\n"
            "- Why this transaction looks like fraud (most suspicious feature)\n"
            "- What is the minimum change to avoid detection\n"
            "- Which feature is the strongest fraud signal\n"
            "- What known fraud pattern does this match\n"
            "- Recommended investigator action"
        )

        try:
            return safe_llm_invoke(
                self.llm,
                [SystemMessage(content=system_prompt), HumanMessage(content=user_message)],
                agent_name=self.agent_name,
                fallback_text="Counterfactual explanation unavailable"
            )
        except Exception as e:
            top_flip = flip_features[0]["feature"] if flip_features else "unknown"
            return (
                f"- Transaction flagged at {orig_prob*100:.1f}% fraud probability "
                f"by stacking ensemble | "
                f"- Strongest perturb-able feature: {top_flip} | "
                f"- {len(flip_features)} feature(s) flip prediction to LEGITIMATE | "
                f"- Automated CF analysis complete — manual review recommended | "
                f"- LLM unavailable: {str(e)[:60]}"
            )

    # ──────────────────────────────────────────────────────────
    # Format CF result for a CSV cell
    # ──────────────────────────────────────────────────────────
    def _format_cf_for_csv(self, cf_result: dict) -> str:
        """
        Serialises counterfactual result to a pipe-delimited string
        safe for a CSV cell.

        Format:
          CF_ANALYSIS|orig_prob=X|n_flips=Y
          || FLIP:feature:orig->flip(direction)   [up to 3 flips]
          || LLM:<narrative with newlines → ' | '>
        """
        if not cf_result:
            return "N/A"

        header = (
            f"CF_ANALYSIS|orig_prob={cf_result['original_probability']}"
            f"|n_flips={cf_result['n_flips_found']}"
        )
        flip_parts = [
            f"FLIP:{f['feature']}:{f['original_value']}"
            f"->{f['flip_value']}({f['change_direction']})"
            for f in cf_result["flip_features"][:3]
        ]
        llm_inline = cf_result["llm_explanation"].replace("\n", " | ")
        return " || ".join([header] + flip_parts + [f"LLM:{llm_inline}"])

    # ──────────────────────────────────────────────────────────
    # Main pipeline entry point
    # ──────────────────────────────────────────────────────────

    def generate_counterfactual_for_transaction(self, visible_context, adjusted_payload):
        signals = visible_context.get("signals", {})
        adjusted_payload = adjusted_payload or {}

        category_rules = [
            {
                "category": "amount",
                "priority": max(
                    self.SIGNAL_PRIORITY.get("extreme_amount_flag", 0) if signals.get("extreme_amount_flag") else 0,
                    self.SIGNAL_PRIORITY.get("very_high_amount_flag", 0) if signals.get("very_high_amount_flag") else 0,
                    self.SIGNAL_PRIORITY.get("high_amount_flag", 0) if signals.get("high_amount_flag") else 0,
                ),
                "active": any(
                    signals.get(k) for k in ["extreme_amount_flag", "very_high_amount_flag", "high_amount_flag"]
                ),
                "suggestion": "Lowering the transaction amount would significantly reduce the risk score.",
            },
            {
                "category": "email_domain",
                "priority": max(
                    self.SIGNAL_PRIORITY.get("suspicious_domain_flag", 0) if signals.get("suspicious_domain_flag") else 0,
                    self.SIGNAL_PRIORITY.get("email_domain_mismatch_flag", 0) if signals.get("email_domain_mismatch_flag") else 0,
                ),
                "active": any(
                    signals.get(k) for k in ["suspicious_domain_flag", "email_domain_mismatch_flag"]
                ),
                "suggestion": (
                    "Using trusted and consistent email domains would reduce domain-related anomaly signals."
                    if signals.get("suspicious_domain_flag") and signals.get("email_domain_mismatch_flag")
                    else (
                        "Using a trusted and commonly used email domain would reduce fraud suspicion."
                        if signals.get("suspicious_domain_flag")
                        else "Aligning sender and receiver email domains would reduce anomaly signals."
                    )
                ),
            },
            {
                "category": "card",
                "priority": self.SIGNAL_PRIORITY.get("rare_card_flag", 0) if signals.get("rare_card_flag") else 0,
                "active": bool(signals.get("rare_card_flag")),
                "suggestion": "Using a common card network would reduce anomaly detection.",
            },
            {
                "category": "address",
                "priority": self.SIGNAL_PRIORITY.get("unusual_addr_flag", 0) if signals.get("unusual_addr_flag") else 0,
                "active": bool(signals.get("unusual_addr_flag")),
                "suggestion": "Providing typical address values would reduce the suspicious profile.",
            },
            {
                "category": "identity",
                "priority": self.SIGNAL_PRIORITY.get("missing_identity_signal", 0) if signals.get("missing_identity_signal") else 0,
                "active": bool(signals.get("missing_identity_signal")),
                "suggestion": "Providing more complete identity information would reduce uncertainty and risk.",
            },
            {
                "category": "timing",
                "priority": self.SIGNAL_PRIORITY.get("night_transaction_flag", 0) if signals.get("night_transaction_flag") else 0,
                "active": bool(signals.get("night_transaction_flag")),
                "suggestion": "Submitting the transaction under less suspicious timing conditions would reduce timing-related risk.",
            },
        ]

        triggered = [
            (rule["priority"], rule["category"], rule["suggestion"])
            for rule in category_rules
            if rule["active"]
        ]
        triggered.sort(key=lambda x: (-x[0], x[1]))

        suggestions = [item[2] for item in triggered[:4]]

        if adjusted_payload.get("signal_count", 0) >= 3 and len(suggestions) < 4:
            suggestions.append(
                "Reducing the number of simultaneous risk signals would likely lower the final verdict severity."
            )

        return {
            "llm_explanation": "\n".join([f"- {s}" for s in suggestions[:4]])
        }
    def run(self, state: dict) -> dict:
        """
        Runs counterfactual interrogation on the top
        N_COUNTERFACTUAL_TRANSACTIONS fraud-flagged transactions.

        Reads from state:
          results_df    — batch predictions DataFrame (from DecisionAgent)
          X_test        — test features (pd.DataFrame, indexed by TransactionID)
          X_train_final / X_train — training data for range computation
          feature_names — ordered list of model feature columns
          meta_learner / xgb_model / lgb_model / cat_model — ensemble models

        Writes to state:
          results_df    — 4 new columns for fraud rows:
                          CounterfactualAnalysis, CF_FlipFeatures,
                          CF_MinChange, CF_LLMExplanation
          cf_report     — full text report string
          cf_summaries  — one-liner per analyzed transaction
        """
        print(f"\n{'='*60}")
        print(f"  [Counterfactual] {self.agent_name} — STARTING")
        print(f"{'='*60}")

        results_df    = state["results_df"].copy()
        X_test        = state["X_test"]
        feature_names = state["feature_names"]
        meta_learner  = state["meta_learner"]
        xgb_model     = state["xgb_model"]
        lgb_model     = state["lgb_model"]
        cat_model     = state["cat_model"]

        # ── Resolve training data for range computation ──────────────
        X_train_raw = state.get("X_train_final", state.get("X_train"))
        if X_train_raw is not None and not isinstance(X_train_raw, pd.DataFrame):
            try:
                X_train_df = pd.DataFrame(X_train_raw, columns=feature_names)
            except Exception:
                X_train_df = None
        else:
            X_train_df = X_train_raw

        # ── Step 1: Perturbation ranges ─────────────────────────────
        print("\n[Step 1/3] Computing feature perturbation ranges...")
        feature_ranges = (
            self._compute_feature_ranges(X_train_df)
            if X_train_df is not None else {}
        )
        if feature_ranges:
            print(f"  [OK] Ranges ready for: {list(feature_ranges.keys())}")
        else:
            print("  [WARN] No training data available — CF will use LLM-only mode")

        # ── Step 2: Select top N fraud transactions ──────────────────
        # Use "TransactionID" — the column written by DecisionAgent
        fraud_rows = (
            results_df[results_df["PredictedLabel"] == 1]
            .nlargest(N_COUNTERFACTUAL_TRANSACTIONS, "FraudProbability")
        )
        print(f"\n[Step 2/3] Analyzing {len(fraud_rows)} highest-probability "
              f"fraud transactions...")

        # Initialise output columns across the full DataFrame
        results_df["CounterfactualAnalysis"] = ""
        results_df["CF_FlipFeatures"]        = ""
        results_df["CF_MinChange"]           = ""
        results_df["CF_LLMExplanation"]      = ""

        cf_summaries = []   # always initialised — safe if fraud_rows is empty

        for i, (row_pos, row_data) in enumerate(fraud_rows.iterrows()):
            txn_id    = row_data["TransactionID"]   # correct column name
            orig_prob = float(row_data["FraudProbability"])
            print(f"  [{i+1}/{len(fraud_rows)}] TransactionID={txn_id}  "
                  f"prob={orig_prob*100:.1f}%")

            # Retrieve original feature values
            transaction = None
            if isinstance(X_test, pd.DataFrame):
                try:
                    transaction = X_test.loc[txn_id].to_dict()
                except KeyError:
                    try:
                        transaction = X_test.iloc[int(txn_id)].to_dict()
                    except Exception:
                        pass

            if transaction is None:
                print(f"    [SKIP] Feature vector not found for TransactionID={txn_id}")
                cf_summaries.append(
                    f"  TransactionID={txn_id} | prob={orig_prob*100:.1f}% | SKIPPED"
                )
                continue

            cf_result = self._analyze_one_transaction(
                txn_index     = txn_id,
                transaction   = transaction,
                orig_prob     = orig_prob,
                feature_names = feature_names,
                feature_ranges= feature_ranges,
                meta_learner  = meta_learner,
                xgb_model     = xgb_model,
                lgb_model     = lgb_model,
                cat_model     = cat_model,
            )

            # Prepare CSV column values
            cf_str     = self._format_cf_for_csv(cf_result)
            flip_names = ", ".join(
                f["feature"] for f in cf_result["flip_features"][:3]
            )
            min_change = ""
            if cf_result["flip_features"]:
                best = cf_result["flip_features"][0]
                min_change = (
                    f"{best['feature']}: "
                    f"{best['original_value']} → {best['flip_value']} "
                    f"({best['change_direction']})"
                )
            llm_inline = cf_result["llm_explanation"].replace("\n", " | ")

            # Write back to the DataFrame
            results_df.loc[row_pos, "CounterfactualAnalysis"] = cf_str
            results_df.loc[row_pos, "CF_FlipFeatures"]        = flip_names
            results_df.loc[row_pos, "CF_MinChange"]           = min_change
            results_df.loc[row_pos, "CF_LLMExplanation"]      = llm_inline

            cf_summaries.append(
                f"  TransactionID={txn_id} | prob={orig_prob*100:.1f}% | "
                f"n_flips={cf_result['n_flips_found']} | "
                f"flip_features=[{flip_names or 'none'}] | "
                f"min_change={min_change or 'N/A'}"
            )

        # ── Step 3: Save report ─────────────────────────────────────
        print(f"\n[Step 3/3] Saving counterfactual report...")
        os.makedirs(REPORTS_DIR, exist_ok=True)

        success_count = sum(1 for s in cf_summaries if "SKIPPED" not in s)

        cf_report = (
            f"FRAUDSENTINEL AI v2.0 — COUNTERFACTUAL INTERROGATION REPORT\n"
            f"{'='*65}\n"
            f"Completed             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Transactions analyzed : {len(fraud_rows)} requested, "
            f"{success_count} successful\n"
            f"Perturb-able features : {len(feature_ranges)}\n"
            f"Steps per feature     : {N_COUNTERFACTUAL_STEPS}\n"
            f"Fraud threshold       : {FRAUD_THRESHOLD}\n\n"
            f"=== FEATURE RANGES USED ===\n"
            + "\n".join(
                f"  {feat}: [{r['min']:.4f}, {r['max']:.4f}]"
                for feat, r in feature_ranges.items()
            )
            + f"\n\n=== PER-TRANSACTION SUMMARY ===\n"
            + "\n".join(cf_summaries)
            + f"\n\n=== CSV COLUMN GUIDE ===\n"
            f"  CounterfactualAnalysis : Full pipe-delimited CF record\n"
            f"    CF_ANALYSIS|orig_prob=X|n_flips=Y "
            f"|| FLIP:feat:orig->flip(dir) || LLM:<narrative>\n\n"
            f"  CF_FlipFeatures : Features that alone flip prediction to LEGIT\n\n"
            f"  CF_MinChange    : Smallest single-feature change achieving a flip\n"
            f"    (i.e. the nearest decision boundary for this transaction)\n\n"
            f"  CF_LLMExplanation : 5-bullet LLaMA 3.3 narrative\n"
            f"    why flagged | fraud pattern | minimum evasion | investigator action\n\n"
            f"=== INTERPRETATION GUIDE ===\n"
            f"  n_flips=0  : Deep in fraud territory. No single-feature\n"
            f"               change can flip the stacking ensemble decision.\n"
            f"               Multiple coordinated changes would be required.\n"
            f"  n_flips>0  : Decision boundary is reachable. First listed\n"
            f"               flip_feature requires the smallest absolute change.\n"
            f"  High orig_prob + n_flips=0 = high-confidence, model-robust fraud.\n"
            f"  High orig_prob + n_flips>0 = flagged but near decision boundary.\n"
        )

        save_report("05_counterfactual_report", cf_report, REPORTS_DIR)
        print(f"  [Report] Saved to reports/05_counterfactual_report.txt")
        print(f"\n  [OK] {self.agent_name} COMPLETE — "
              f"{success_count}/{len(fraud_rows)} transactions analyzed successfully")

        return {
            **state,
            "results_df"  : results_df,
            "cf_report"   : cf_report,
            "cf_summaries": cf_summaries,
            "status"      : "counterfactual_complete",
            "agent"       : self.agent_name,
        }
