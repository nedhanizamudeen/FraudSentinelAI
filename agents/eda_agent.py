# -------------------------------------------------------------
#  FraudSentinel AI — agents/eda_agent.py
#
#  AGENT 1: EDA Agent (Exploratory Data Analysis Agent)
# -------------------------------------------------------------

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from tools.data_tools import (
    load_and_merge_data,
    analyze_missing_values,
    analyze_class_balance,
    get_column_types,
    undersample_majority,
    save_report,
)
from config import GROQ_API_KEY, LLM_MODEL, REPORTS_DIR, TARGET_CLASS_RATIO
from tools.safe_llm import safe_llm_invoke
import os


class EDAAgent:
    """
    The EDA Agent is the first agent to run in FraudSentinel AI.

    It:
    1. Loads and merges the IEEE-CIS dataset
    2. Analyzes missing values
    3. Checks class imbalance
    4. Identifies column types
    5. Uses the LLM to reason about findings and make decisions
    6. Passes a clean analysis to the Feature Engineering Agent
    """

    def __init__(self):
        # Initialize the LLM brain (Groq + LLaMA3)
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=LLM_MODEL,
            temperature=0,
        )
        self.agent_name = "EDA Agent"

    def _think(self, data_summary: str) -> str:
        """
        Safely reason about EDA findings using the LLM.
        Falls back to template text if LLM is unavailable.
        """
        system_prompt = """You are the EDA Agent in FraudSentinel AI.
Respond in exactly 4 short bullet points. Each bullet = one sentence, max 20 words.
No headers, no paragraphs. Just 4 clean bullets starting with -"""

        user_message = f"""Dataset stats (after undersampling non-fraud to 1:{TARGET_CLASS_RATIO} ratio):
{data_summary}

Give exactly 4 bullet points:
- Most important finding
- Impact of reducing class ratio from 1:27 to 1:{TARGET_CLASS_RATIO}
- Key action for Feature Engineering Agent
- Dataset difficulty (one sentence)"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        fallback_text = (
            "- Dataset remains highly challenging due to fraud rarity and many missing values.\n"
            f"- Reducing imbalance to 1:{TARGET_CLASS_RATIO} improves learnability while preserving fraud signal.\n"
            "- Feature engineering should focus on robust encoding, missingness handling, and column reduction.\n"
            "- Overall difficulty is moderate-to-high due to heterogeneous features and class imbalance."
        )

        return safe_llm_invoke(
            self.llm,
            messages,
            agent_name=self.agent_name,
            fallback_text=fallback_text,
        )

    def run(self, state: dict) -> dict:
        """
        Main execution method of the EDA Agent.
        """
        print(f"\n{'=' * 60}")
        print(f"  [EDA] {self.agent_name} — STARTING")
        print(f"{'=' * 60}")

        # -- STEP 1: Load and merge data --
        print("\n[Step 1/4] Loading IEEE-CIS dataset...")
        load_result = load_and_merge_data()

        if load_result["status"] == "error":
            print(f"  [ERROR] ERROR: {load_result['summary']}")
            return {
                **state,
                "error": load_result["summary"],
                "status": "failed",
                "agent": self.agent_name,
            }

        df = load_result["data"]
        all_summaries = ["=== DATA LOADING ===\n" + load_result["summary"]]

        # -- STEP 1b: Undersample majority class to 1:10 ratio --
        print(f"\n[Step 1b/4] Undersampling non-fraud to 1:{TARGET_CLASS_RATIO} ratio...")
        undersample_result = undersample_majority(df)
        df = undersample_result["data"]
        all_summaries.append(
            "=== MAJORITY UNDERSAMPLING ===\n" + undersample_result["summary"]
        )
        print(f"  [OK] Dataset reduced from original 1:27 -> 1:{TARGET_CLASS_RATIO} ratio")

        # -- STEP 2: Analyze missing values --
        print("\n[Step 2/4] Analyzing missing values...")
        missing_result = analyze_missing_values(df)
        all_summaries.append("=== MISSING VALUES ===\n" + missing_result["summary"])
        print(f"  Found {len(missing_result['columns_to_drop'])} columns to drop")

        # -- STEP 3: Check class balance --
        print("\n[Step 3/4] Checking class balance...")
        balance_result = analyze_class_balance(df)
        all_summaries.append("=== CLASS BALANCE ===\n" + balance_result["summary"])

        # -- STEP 4: Identify column types --
        print("\n[Step 4/4] Identifying column types...")
        type_result = get_column_types(df)
        all_summaries.append("=== COLUMN TYPES ===\n" + type_result["summary"])

        # -- LLM REASONING --
        print("\n[LLM] EDA Agent is reasoning about findings...")
        combined_summary = "\n\n".join(all_summaries)
        llm_analysis = self._think(combined_summary)

        short_lines = [l.strip() for l in llm_analysis.split("\n") if l.strip()][:6]
        print(f"\n  [LLM] LLM Decision:")
        for line in short_lines:
            print(f"     {line}")

        # -- Save EDA report --
        full_report = (
            f"FRAUDSENTINEL AI — EDA AGENT REPORT\n"
            f"{'=' * 50}\n\n"
            f"{combined_summary}\n\n"
            f"{'=' * 50}\n"
            f"LLM REASONING:\n{llm_analysis}"
        )
        report_path = save_report("01_eda_report", full_report, REPORTS_DIR)
        print(f"  [Report] Report saved: {report_path}")

        print(f"\n  [OK] {self.agent_name} COMPLETE")

        return {
            **state,
            "df": df,
            "columns_to_drop": missing_result["columns_to_drop"],
            "is_imbalanced": balance_result["is_imbalanced"],
            "fraud_ratio": balance_result["fraud_ratio"],
            "numerical_cols": type_result["numerical_cols"],
            "categorical_cols": type_result["categorical_cols"],
            "eda_summary": combined_summary,
            "eda_llm_analysis": llm_analysis,
            "status": "eda_complete",
            "agent": self.agent_name,
        }