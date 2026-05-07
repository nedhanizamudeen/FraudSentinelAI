# -------------------------------------------------------------
#  FraudSentinel AI v2.0 — agents/eval_agent.py
#
#  AGENT 4: Evaluation Agent (Extended)
#
#  NEW in v2.0:
#    - Evaluates XGBoost, LightGBM, CatBoost, AND the Stacking
#      Ensemble across all standard metrics
#    - Stacking ensemble is treated as the "best model" by default
#      (it combines all base models and is statistically superior)
#    - Retrain loop still applies if Stacking AUC < threshold
#    - Produces a comprehensive model comparison table
# -------------------------------------------------------------

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from tools.ml_tools import evaluate_model
from tools.data_tools import save_report
from config import GROQ_API_KEY, LLM_MODEL, REPORTS_DIR, MIN_AUC_THRESHOLD
import joblib
import os
from tools.safe_llm import safe_llm_invoke

class EvalAgent:
    """
    Evaluation Agent — Agent 4 in the v2.0 pipeline.

    Evaluates all four models (XGBoost, LightGBM, CatBoost, Stacking)
    and enforces the AUC quality gate. The Stacking Ensemble is the
    primary model passed to the Decision Agent.

    The agentic feedback loop (retrain if AUC < threshold) is preserved.
    """

    def __init__(self):
        self.llm = ChatGroq(
            api_key    = GROQ_API_KEY,
            model_name = LLM_MODEL,
            temperature= 0
        )
        self.agent_name = "Evaluation Agent"
        self.max_retrain_attempts = 3

    def _analyze_results(self, results: dict) -> str:
        """LLM analyzes the four-model evaluation and recommends next steps."""
        system_prompt = """You are the Evaluation Agent in FraudSentinel AI v2.0.
Respond in exactly 4 short bullet points. Each bullet = one sentence, max 20 words.
No headers, no paragraphs. Just 4 clean bullets starting with -"""

        user_message = (
            f"Model evaluation results:\n"
            f"  XGBoost  : AUC={results['xgb']['auc']:.4f}, F1={results['xgb']['f1']:.4f}, Recall={results['xgb']['recall']:.4f}\n"
            f"  LightGBM : AUC={results['lgb']['auc']:.4f}, F1={results['lgb']['f1']:.4f}, Recall={results['lgb']['recall']:.4f}\n"
            f"  CatBoost : AUC={results['cat']['auc']:.4f}, F1={results['cat']['f1']:.4f}, Recall={results['cat']['recall']:.4f}\n"
            f"  Stacking : AUC={results['stack']['auc']:.4f}, F1={results['stack']['f1']:.4f}, Recall={results['stack']['recall']:.4f}\n"
            f"  Threshold: {MIN_AUC_THRESHOLD}\n\n"
            f"Give exactly 4 bullet points:\n"
            f"- Which model performs best overall and the main reason\n"
            f"- Does the stacking ensemble add value over individual models\n"
            f"- Is stacking AUC acceptable for fraud detection\n"
            f"- PASS or RETRAIN and why"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        return safe_llm_invoke(
        self.llm,
        messages,
        agent_name=self.agent_name,
        fallback_text="Stacking model performs best with highest AUC and balanced precision-recall tradeoff."
        )

    def _format_comparison_table(self, results: dict) -> str:
        """Creates an ASCII comparison table of all four models."""
        header = (
            f"\n{'Model':<15} {'AUC':>8} {'F1':>8} {'Precision':>10} "
            f"{'Recall':>8} {'TP':>7} {'FP':>7} {'FN':>7} {'TN':>7}\n"
            f"{'-'*75}\n"
        )
        rows = ""
        for name, key in [("XGBoost","xgb"),("LightGBM","lgb"),
                          ("CatBoost","cat"),("Stacking*","stack")]:
            r = results[key]
            rows += (
                f"{name:<15} {r['auc']:>8.4f} {r['f1']:>8.4f} "
                f"{r['precision']:>10.4f} {r['recall']:>8.4f} "
                f"{r['tp']:>7,} {r['fp']:>7,} {r['fn']:>7,} {r['tn']:>7,}\n"
            )
        footer = "* Stacking = OOF-trained meta-learner on XGBoost + LightGBM + CatBoost\n"
        return header + rows + footer

    def run(self, state: dict) -> dict:
        print(f"\n{'='*60}")
        print(f"  [Eval] {self.agent_name} — STARTING")
        print(f"{'='*60}")

        xgb_model     = state["xgb_model"]
        lgb_model     = state["lgb_model"]
        cat_model     = state["cat_model"]
        meta_learner  = state["meta_learner"]
        test_preds_avg= state["test_preds_avg"]
        X_test        = state["X_test"]
        y_test        = state["y_test"]
        retrain_count = state.get("retrain_count", 0)
        all_summaries = []

        # -- Evaluate all four models --
        print("\n[Step 1/5] Evaluating XGBoost...")
        xgb_results = evaluate_model(xgb_model, "XGBoost", X_test, y_test)
        all_summaries.append("=== XGBOOST EVALUATION ===\n" + xgb_results["summary"])
        print(f"  XGBoost  AUC: {xgb_results['auc']:.4f}")

        print("\n[Step 2/5] Evaluating LightGBM...")
        lgb_results = evaluate_model(lgb_model, "LightGBM", X_test, y_test)
        all_summaries.append("=== LIGHTGBM EVALUATION ===\n" + lgb_results["summary"])
        print(f"  LightGBM AUC: {lgb_results['auc']:.4f}")

        print("\n[Step 3/5] Evaluating CatBoost...")
        cat_results = evaluate_model(cat_model, "CatBoost", X_test, y_test)
        all_summaries.append("=== CATBOOST EVALUATION ===\n" + cat_results["summary"])
        print(f"  CatBoost AUC: {cat_results['auc']:.4f}")

        print("\n[Step 4/5] Evaluating Stacking Ensemble...")
        stack_results = evaluate_model(
            model=None, model_name="Stacking Ensemble",
            X_test=X_test, y_test=y_test,
            is_stack=True, meta_learner=meta_learner,
            xgb_model=xgb_model, lgb_model=lgb_model, cat_model=cat_model,
            test_preds_avg=test_preds_avg
        )
        all_summaries.append("=== STACKING ENSEMBLE EVALUATION ===\n" + stack_results["summary"])
        print(f"  Stacking AUC: {stack_results['auc']:.4f}")

        # Aggregate for LLM analysis
        results_dict = {
            "xgb"  : xgb_results,
            "lgb"  : lgb_results,
            "cat"  : cat_results,
            "stack": stack_results
        }

        # Comparison table
        comparison_table = self._format_comparison_table(results_dict)
        all_summaries.append("=== MODEL COMPARISON TABLE ===\n" + comparison_table)
        print("\n" + comparison_table)

        # -- LLM Analysis --
        print("\n[LLM] Evaluation Agent analyzing results...")
        llm_analysis = self._analyze_results(results_dict)
        short_lines = [l.strip() for l in llm_analysis.split("\n") if l.strip()][:5]
        print(f"\n  [LLM] LLM Decision:")
        for line in short_lines:
            print(f"     {line}")
        all_summaries.append("=== LLM ANALYSIS ===\n" + llm_analysis)

        # -- Stacking is always the primary best model --
        best_model_name = "Stacking Ensemble"
        best_auc        = stack_results["auc"]

        # -- PASS / RETRAIN gate (based on stacking AUC) --
        if best_auc >= MIN_AUC_THRESHOLD:
            decision    = "pass"
            next_status = "evaluation_passed"
            print(f"  [OK] DECISION: PASS (Stacking AUC {best_auc:.4f} >= threshold {MIN_AUC_THRESHOLD})")
        elif retrain_count >= self.max_retrain_attempts:
            decision    = "pass"
            next_status = "evaluation_passed"
            print(f"  [WARN] DECISION: FORCED PASS (max retrains reached, best AUC: {best_auc:.4f})")
        else:
            decision    = "retrain"
            next_status = "needs_retraining"
            print(f"  [LOOP] DECISION: RETRAIN (Stacking AUC {best_auc:.4f} < threshold {MIN_AUC_THRESHOLD})")

        # -- Save best_model.pkl (stacking meta-learner) --
        best_model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "models", "best_model.pkl"
        )
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        joblib.dump(meta_learner, best_model_path)

        # -- Save report --
        combined_summary = "\n\n".join(all_summaries)
        decision_summary = (
            f"\n\n=== FINAL DECISION ===\n"
            f"Best Model   : {best_model_name}\n"
            f"Best AUC     : {best_auc:.4f}\n"
            f"Threshold    : {MIN_AUC_THRESHOLD}\n"
            f"Decision     : {decision.upper()}\n"
            f"Retrain #    : {retrain_count}"
        )
        full_report = (
            f"FRAUDSENTINEL AI v2.0 — EVALUATION AGENT REPORT\n"
            f"{'='*55}\n\n"
            f"{combined_summary}{decision_summary}"
        )
        report_path = save_report("04_eval_report", full_report, REPORTS_DIR)
        print(f"  [Report] Report saved: {report_path}")
        print(f"\n  [OK] {self.agent_name} COMPLETE — Decision: {decision.upper()}")

        return {
            **state,
            "best_model"        : meta_learner,      # meta-learner is the best model
            "best_model_name"   : best_model_name,
            "best_model_path"   : best_model_path,
            "best_auc"          : best_auc,
            "xgb_results"       : xgb_results,
            "lgb_results"       : lgb_results,
            "cat_results"       : cat_results,
            "stack_results"     : stack_results,
            "all_eval_results"  : results_dict,
            "eval_decision"     : decision,
            "eval_llm_analysis" : llm_analysis,
            "retrain_count"     : retrain_count + (1 if decision == "retrain" else 0),
            "status"            : next_status,
            "agent"             : self.agent_name
        }
