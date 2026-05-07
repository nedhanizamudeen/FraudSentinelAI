# -------------------------------------------------------------
#  FraudSentinel AI v2.0 — agents/model_agent.py
#
#  AGENT 3: Model Training Agent (Extended)
#
#  NEW in v2.0:
#    - Trains CatBoost in addition to XGBoost and LightGBM
#    - Builds a TRUE stacking ensemble using OOF predictions
#      (prevents data leakage — each training row is predicted
#      by a model that did NOT see it during training)
#    - The meta-learner (Logistic Regression) learns how to
#      combine the three base models optimally
# -------------------------------------------------------------

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from tools.ml_tools import (
    apply_smote,
    train_xgboost,
    train_lightgbm,
    train_catboost,
    build_stacking_ensemble
)
from tools.data_tools import save_report
from config import GROQ_API_KEY, LLM_MODEL, REPORTS_DIR
from tools.safe_llm import safe_llm_invoke

class ModelAgent:
    """
    The Model Training Agent — Agent 3 in the v2.0 pipeline.

    Extended from v1.0:
      v1.0: XGBoost + LightGBM
      v2.0: XGBoost + LightGBM + CatBoost + Stacking Ensemble

    The stacking ensemble uses out-of-fold (OOF) predictions to
    train the meta-learner, strictly preventing data leakage.
    """

    def __init__(self):
        self.llm = ChatGroq(
            api_key    = GROQ_API_KEY,
            model_name = LLM_MODEL,
            temperature= 0
        )
        self.agent_name = "Model Training Agent"

    def _decide_strategy(self, state: dict) -> str:
        """LLM decides the training strategy."""
        system_prompt = """You are the Model Training Agent in FraudSentinel AI v2.0.
Respond in exactly 4 short bullet points. Each bullet = one sentence, max 20 words.
No headers, no paragraphs. Just 4 clean bullets starting with -"""

        user_message = (
            f"Dataset: imbalanced={state.get('is_imbalanced', True)}, "
            f"fraud={state.get('fraud_ratio', 3.5):.1f}%, "
            f"features={len(state.get('feature_names', []))}\n\n"
            f"We train XGBoost, LightGBM, CatBoost, then stack with OOF.\n\n"
            f"Give exactly 4 bullet points:\n"
            f"- Should we apply SMOTE and why\n"
            f"- Which base model will likely perform best and why\n"
            f"- Key benefit of OOF stacking for this fraud dataset\n"
            f"- One special consideration for the stacking meta-learner"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        return safe_llm_invoke(
        self.llm,
        messages,
        agent_name=self.agent_name,
        fallback_text="Using standard ensemble strategy: XGBoost, LightGBM, CatBoost, and Stacking."
        )

    def run(self, state: dict) -> dict:
        """
        Main execution: trains XGBoost, LightGBM, CatBoost,
        then builds the stacking ensemble.
        """
        print(f"\n{'='*60}")
        print(f"  [Model] {self.agent_name} — STARTING")
        print(f"{'='*60}")

        X_train       = state["X_train"]
        y_train       = state["y_train"]
        X_test        = state["X_test"]
        is_imbalanced = state.get("is_imbalanced", True)
        all_summaries = []

        # -- LLM DECIDES STRATEGY --
        print("\n[LLM] Model Agent deciding training strategy...")
        strategy = self._decide_strategy(state)
        short_lines = [l.strip() for l in strategy.split("\n") if l.strip()][:5]
        print(f"\n  [LLM] LLM Decision:")
        for line in short_lines:
            print(f"     {line}")
        all_summaries.append("=== LLM STRATEGY DECISION ===\n" + strategy)

        # -- STEP 1: Apply SMOTE --
        X_train_final = X_train
        y_train_final = y_train

        if is_imbalanced:
            print("\n[Step 1/5] Applying SMOTE (dataset is imbalanced)...")
            smote_result  = apply_smote(X_train, y_train)
            X_train_final = smote_result["X_train_resampled"]
            y_train_final = smote_result["y_train_resampled"]
            all_summaries.append("=== SMOTE ===\n" + smote_result["summary"])
            print(f"  [OK] SMOTE applied successfully")
        else:
            print("\n[Step 1/5] Skipping SMOTE (dataset is balanced)")
            all_summaries.append("=== SMOTE ===\nSMOTE skipped — dataset is already balanced.")

        # -- STEP 2: Train XGBoost --
        print("\n[Step 2/5] Training XGBoost...")
        xgb_result = train_xgboost(X_train_final, y_train_final)
        all_summaries.append("=== XGBOOST TRAINING ===\n" + xgb_result["summary"])
        print(f"  [OK] XGBoost trained and saved")

        # -- STEP 3: Train LightGBM --
        print("\n[Step 3/5] Training LightGBM...")
        lgb_result = train_lightgbm(X_train_final, y_train_final)
        all_summaries.append("=== LIGHTGBM TRAINING ===\n" + lgb_result["summary"])
        print(f"  [OK] LightGBM trained and saved")

        # -- STEP 4: Train CatBoost --
        print("\n[Step 4/5] Training CatBoost...")
        cat_result = train_catboost(X_train_final, y_train_final)
        all_summaries.append("=== CATBOOST TRAINING ===\n" + cat_result["summary"])
        print(f"  [OK] CatBoost trained and saved")

        # -- STEP 5: Build Stacking Ensemble with OOF predictions --
        print("\n[Step 5/5] Building Stacking Ensemble (OOF — no data leakage)...")
        stack_result = build_stacking_ensemble(
            X_train       = X_train_final,
            y_train       = y_train_final,
            xgb_model     = xgb_result["model"],
            lgb_model     = lgb_result["model"],
            cat_model     = cat_result["model"],
            X_test        = X_test
        )
        all_summaries.append("=== STACKING ENSEMBLE ===\n" + stack_result["summary"])
        print(f"  [OK] Stacking ensemble built (Meta OOF AUC: {stack_result['oof_auc_meta']:.4f})")

        # -- Save report --
        combined_summary = "\n\n".join(all_summaries)
        full_report = (
            f"FRAUDSENTINEL AI v2.0 — MODEL TRAINING AGENT REPORT\n"
            f"{'='*55}\n\n"
            f"{combined_summary}"
        )
        report_path = save_report("03_model_report", full_report, REPORTS_DIR)
        print(f"  [Report] Report saved: {report_path}")

        print(f"\n  [OK] {self.agent_name} COMPLETE")

        return {
            **state,
            "xgb_model"         : xgb_result["model"],
            "lgb_model"         : lgb_result["model"],
            "cat_model"         : cat_result["model"],
            "meta_learner"      : stack_result["meta_learner"],
            "test_preds_avg"    : stack_result["test_preds_avg"],
            "oof_preds"         : stack_result["oof_preds"],
            "oof_auc_xgb"       : stack_result["oof_auc_xgb"],
            "oof_auc_lgb"       : stack_result["oof_auc_lgb"],
            "oof_auc_cat"       : stack_result["oof_auc_cat"],
            "oof_auc_meta"      : stack_result["oof_auc_meta"],
            "trained_models"    : {
                "XGBoost" : xgb_result["model"],
                "LightGBM": lgb_result["model"],
                "CatBoost": cat_result["model"],
                "Stacking": stack_result["meta_learner"]
            },
            "X_train_final"     : X_train_final,
            "y_train_final"     : y_train_final,
            "model_strategy"    : strategy,
            "training_summary"  : combined_summary,
            "status"            : "models_trained",
            "agent"             : self.agent_name,
            "retrain_count"     : state.get("retrain_count", 0)
        }
