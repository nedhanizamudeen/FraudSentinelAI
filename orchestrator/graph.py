# orchestrator/graph.py
# -------------------------------------------------------------
#  FraudSentinel AI v2.0 — orchestrator/graph.py
# -------------------------------------------------------------

from langgraph.graph import StateGraph, END
from typing import TypedDict, Any, Optional

from agents.eda_agent import EDAAgent
from agents.feature_agent import FeatureAgent
from agents.model_agent import ModelAgent
from agents.eval_agent import EvalAgent
from agents.decision_agent import DecisionAgent


class FraudSentinelState(TypedDict, total=False):
    df: Any
    columns_to_drop: list
    is_imbalanced: bool
    fraud_ratio: float
    numerical_cols: list
    categorical_cols: list
    eda_summary: str
    eda_llm_analysis: str

    df_processed: Any
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    feature_names: list
    new_features: list
    encoders: dict
    feature_summary: str
    feature_llm_analysis: str
    feature_pipeline_path: str

    xgb_model: Any
    lgb_model: Any
    cat_model: Any
    meta_learner: Any
    oof_preds: Any
    test_preds_avg: Any
    oof_auc_xgb: float
    oof_auc_lgb: float
    oof_auc_cat: float
    oof_auc_meta: float
    trained_models: dict
    X_train_final: Any
    y_train_final: Any
    model_strategy: str
    training_summary: str
    retrain_count: int

    best_model: Any
    best_model_name: str
    best_model_path: str
    best_auc: float
    xgb_results: dict
    lgb_results: dict
    cat_results: dict
    stack_results: dict
    all_eval_results: dict
    eval_decision: str
    eval_llm_analysis: str

    results_df: Any
    fraud_df: Any
    legit_df: Any
    csv_path: str
    fraud_csv_path: str
    legit_csv_path: str
    cm_plot_path: str
    final_report: str
    batch_stats: dict
    cf_report: str
    cf_summaries: list

    status: str
    agent: str
    error: Optional[str]


_eda_agent = EDAAgent()
_feature_agent = FeatureAgent()
_model_agent = ModelAgent()
_eval_agent = EvalAgent()
_decision_agent = DecisionAgent()


def run_eda(state: FraudSentinelState) -> FraudSentinelState:
    return _eda_agent.run(state)


def run_features(state: FraudSentinelState) -> FraudSentinelState:
    return _feature_agent.run(state)


def run_model(state: FraudSentinelState) -> FraudSentinelState:
    return _model_agent.run(state)


def run_eval(state: FraudSentinelState) -> FraudSentinelState:
    return _eval_agent.run(state)


def run_decision(state: FraudSentinelState) -> FraudSentinelState:
    return _decision_agent.run(state)


def should_retrain(state: FraudSentinelState) -> str:
    decision = state.get("eval_decision", "pass")
    if decision == "retrain":
        print("\n  [LOOP] ORCHESTRATOR: Sending back to Model Agent for retraining...")
        return "retrain"
    print("\n  [OK] ORCHESTRATOR: Evaluation passed.")
    return "proceed"


def build_graph(training_only: bool = False):
    graph = StateGraph(FraudSentinelState)

    graph.add_node("eda_agent", run_eda)
    graph.add_node("feature_agent", run_features)
    graph.add_node("model_agent", run_model)
    graph.add_node("eval_agent", run_eval)
    if not training_only:
        graph.add_node("decision_agent", run_decision)

    graph.add_edge("eda_agent", "feature_agent")
    graph.add_edge("feature_agent", "model_agent")
    graph.add_edge("model_agent", "eval_agent")

    if training_only:
        graph.add_conditional_edges(
            source="eval_agent",
            path=should_retrain,
            path_map={
                "retrain": "model_agent",
                "proceed": END,
            },
        )
    else:
        graph.add_conditional_edges(
            source="eval_agent",
            path=should_retrain,
            path_map={
                "retrain": "model_agent",
                "proceed": "decision_agent",
            },
        )
        graph.add_edge("decision_agent", END)

    graph.set_entry_point("eda_agent")
    return graph.compile()


def run_pipeline(training_only: bool = False) -> dict:
    print(f"\n{'='*70}")
    if training_only:
        print("  [FraudSentinel] FRAUDSENTINEL AI v2.2 — TRAINING PIPELINE STARTING")
        print("  Agents: EDA → Feature → Model(XGB+LGB+CAT+Stack) → Eval")
        print("  Decision/Counterfactual generation is skipped during training mode")
    else:
        print("  [FraudSentinel] FRAUDSENTINEL AI v2.2 — FULL PIPELINE STARTING")
        print("  Agents: EDA → Feature → Model(XGB+LGB+CAT+Stack) → Eval → Decision+CF")
    print(f"{'='*70}\n")

    graph = build_graph(training_only=training_only)
    initial_state = {"status": "starting", "retrain_count": 0}
    final_state = graph.invoke(initial_state)
    return final_state