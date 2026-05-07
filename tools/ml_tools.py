# -------------------------------------------------------------
#  FraudSentinel AI v2.0 — tools/ml_tools.py
#
#  Extended ML toolkit:
#    - Original tools: clean, encode, engineer, split, SMOTE,
#      XGBoost, LightGBM, evaluate, predict
#    - NEW: CatBoost training
#    - NEW: True Stacking Ensemble with Out-of-Fold (OOF)
#           predictions to prevent data leakage
#    - NEW: Stacking meta-learner prediction
# -------------------------------------------------------------

import pandas as pd
import numpy as np
import joblib
import os
import warnings
import hashlib
import json
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from config import (
    TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, TARGET_CLASS_RATIO,
    MODELS_DIR, MAX_MISSING_RATIO,
    N_STACK_FOLDS, META_LEARNER, FEATURE_PIPELINE_PATH, TRAINING_METADATA_PATH,
    REQUIRED_TRAINED_ARTIFACTS,
    SINGLE_TRANSACTION_INPUT_SCHEMA,
    SIMPLIFIED_TRANSACTION_INPUT_SCHEMA,
    LEGACY_TRANSACTION_INPUT_SCHEMA,
    COMMON_EMAIL_DOMAINS,
    SUSPICIOUS_EMAIL_DOMAINS,
    COMMON_CARD_NETWORKS,
    HIGH_AMOUNT_THRESHOLD, VERY_HIGH_AMOUNT_THRESHOLD, EXTREME_AMOUNT_THRESHOLD,
    RISK_ADJUSTMENT_WEIGHTS,
)
import hashlib


# ==========================================
#  TOOL 1 — Clean and Prepare Data
# ==========================================
def clean_data(df: pd.DataFrame, columns_to_drop: list) -> dict:
    """
    Cleans the raw merged dataframe:
      1. Drops high-missing columns
      2. Drops TransactionID
      3. Fills missing values (median for numeric, 'Unknown' for text)
    """
    print("  [MLTool] Dropping high-missing columns...")
    df_clean = df.drop(columns=columns_to_drop + ["TransactionID"],
                       errors="ignore")

    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COLUMN in num_cols:
        num_cols.remove(TARGET_COLUMN)

    for col in num_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    cat_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        df_clean[col] = df_clean[col].fillna("Unknown")

    summary = (
        f"Data Cleaning Complete:\n"
        f"  Dropped {len(columns_to_drop)} high-missing columns\n"
        f"  Filled {len(num_cols)} numerical columns with median\n"
        f"  Filled {len(cat_cols)} categorical columns with 'Unknown'\n"
        f"  Final shape: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns"
    )
    return {"data": df_clean, "summary": summary}


# ==========================================
#  TOOL 2 — Encode Categorical Columns
# ==========================================
def encode_categoricals(df: pd.DataFrame) -> dict:
    """Converts text columns to numbers using Label Encoding."""
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    print(f"  [MLTool] Encoding {len(cat_cols)} categorical columns...")

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    summary = (
        f"Categorical Encoding Complete:\n"
        f"  Encoded {len(cat_cols)} columns using Label Encoding\n"
        f"  Sample columns: {cat_cols[:5]}"
    )
    return {"data": df, "encoded_columns": cat_cols, "encoders": encoders, "summary": summary}


# ==========================================
#  TOOL 3 — Engineer New Features
# ==========================================
def engineer_features(df: pd.DataFrame) -> dict:
    """Creates new fraud-detection features."""
    print("  [MLTool] Engineering new features...")
    df = df.copy()
    new_features = []

    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
        new_features.append("TransactionAmt_log")

    if all(c in df.columns for c in ["TransactionAmt", "card1"]):
        card_mean = df.groupby("card1")["TransactionAmt"].transform("mean")
        df["TransactionAmt_to_card_mean"] = df["TransactionAmt"] / (card_mean + 1)
        new_features.append("TransactionAmt_to_card_mean")

    if "TransactionDT" in df.columns:
        df["transaction_hour"] = (df["TransactionDT"] // 3600) % 24
        df["transaction_day"]  = (df["TransactionDT"] // 86400) % 7
        new_features.extend(["transaction_hour", "transaction_day"])

    if "transaction_hour" in df.columns:
        df["is_night_transaction"] = (
            (df["transaction_hour"] >= 0) & (df["transaction_hour"] <= 5)
        ).astype(int)
        new_features.append("is_night_transaction")

    if "card1" in df.columns:
        df["card1_count"] = df.groupby("card1")["card1"].transform("count")
        new_features.append("card1_count")

    if all(c in df.columns for c in ["TransactionAmt", "addr1"]):
        addr_mean = df.groupby("addr1")["TransactionAmt"].transform("mean")
        df["amt_addr_ratio"] = df["TransactionAmt"] / (addr_mean + 1)
        new_features.append("amt_addr_ratio")

    summary = (
        f"Feature Engineering Complete:\n"
        f"  Created {len(new_features)} new features:\n"
    )
    for f in new_features:
        summary += f"    + {f}\n"

    return {"data": df, "new_features": new_features, "summary": summary}


# ==========================================
#  TOOL 4 — Split Data
# ==========================================
def split_data(df: pd.DataFrame) -> dict:
    """Splits data into train/test sets (80/20, stratified)."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    summary = (
        f"Data Split Complete:\n"
        f"  Training set : {X_train.shape[0]:,} rows ({(1-TEST_SIZE)*100:.0f}%)\n"
        f"  Testing set  : {X_test.shape[0]:,} rows ({TEST_SIZE*100:.0f}%)\n"
        f"  Features     : {X_train.shape[1]} columns\n"
        f"  Fraud in train: {y_train.sum():,} ({y_train.mean()*100:.2f}%)"
    )
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": list(X.columns),
        "summary": summary
    }


# ==========================================
#  TOOL 5 — Apply SMOTE
# ==========================================
def apply_smote(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Applies SMOTE to balance the training set."""
    print("  [MLTool] Applying SMOTE to balance training data...")
    print(f"    Before: {y_train.value_counts().to_dict()}")

    smote = SMOTE(random_state=RANDOM_STATE, n_jobs=-1)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"    After : {pd.Series(y_resampled).value_counts().to_dict()}")
    synthesized = int(y_resampled.sum()) - int(y_train.sum())

    summary = (
        f"SMOTE Applied (after training-only undersampling):\n"
        f"  Before : {y_train.sum():,} fraud / {(y_train==0).sum():,} non-fraud\n"
        f"  After  : {y_resampled.sum():,} fraud / {(y_resampled==0).sum():,} non-fraud\n"
        f"  Synthetic Class-1 rows created : {synthesized:,}\n"
        f"  Training set is now perfectly balanced!"
    )
    return {
        "X_train_resampled": X_resampled,
        "y_train_resampled": y_resampled,
        "summary": summary
    }




def undersample_training_data(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Undersample only the training split so the test set keeps the natural distribution."""
    fraud_mask = y_train == 1
    non_fraud_mask = y_train == 0
    fraud_idx = y_train[fraud_mask].index
    non_fraud_idx = y_train[non_fraud_mask].index

    fraud_count = len(fraud_idx)
    original_non_fraud = len(non_fraud_idx)
    target_non_fraud = fraud_count * TARGET_CLASS_RATIO

    if fraud_count == 0 or original_non_fraud <= target_non_fraud:
        summary = (
            f"Training undersampling skipped. Fraud={fraud_count:,}, non-fraud={original_non_fraud:,}."
        )
        return {"X_train": X_train, "y_train": y_train, "summary": summary}

    sampled_non_fraud = pd.Series(non_fraud_idx).sample(n=target_non_fraud, random_state=RANDOM_STATE).tolist()
    keep_idx = list(fraud_idx) + sampled_non_fraud
    X_out = X_train.loc[keep_idx].sample(frac=1, random_state=RANDOM_STATE)
    y_out = y_train.loc[X_out.index]
    summary = (
        f"Training-only majority undersampling complete:\n"
        f"  Fraud rows kept      : {fraud_count:,}\n"
        f"  Non-fraud BEFORE     : {original_non_fraud:,}\n"
        f"  Non-fraud AFTER      : {target_non_fraud:,}\n"
        f"  Test split preserved : natural distribution unchanged"
    )
    return {"X_train": X_out, "y_train": y_out, "summary": summary}


def save_feature_pipeline(pipeline: dict, path: str = FEATURE_PIPELINE_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_feature_pipeline(path: str = FEATURE_PIPELINE_PATH) -> dict:
    return joblib.load(path)


def _apply_saved_label_encoding(df: pd.DataFrame, encoder_classes: dict) -> pd.DataFrame:
    df = df.copy()
    for col, classes in encoder_classes.items():
        if col not in df.columns:
            df[col] = "Unknown"
        series = df[col].astype(str)
        mapping = {str(v): i for i, v in enumerate(classes)}
        unknown_idx = mapping.get("Unknown", -1)
        df[col] = series.map(lambda x: mapping.get(x, unknown_idx)).astype(float)
    return df

def normalize_email_domain(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    s = str(value).strip().lower()
    if not s:
        return None

    if "@" in s:
        s = s.split("@")[-1].strip()

    s = s.replace(" ", "")
    return s if s else None


def classify_email_domain(domain: str | None) -> dict:
    normalized = normalize_email_domain(domain)

    if normalized is None:
        return {
            "normalized": None,
            "is_common_domain": 0,
            "is_suspicious_domain": 0,
            "domain_risk_tier": "unknown",
        }

    if normalized in COMMON_EMAIL_DOMAINS:
        return {
            "normalized": normalized,
            "is_common_domain": 1,
            "is_suspicious_domain": 0,
            "domain_risk_tier": "low",
        }

    if normalized in SUSPICIOUS_EMAIL_DOMAINS:
        return {
            "normalized": normalized,
            "is_common_domain": 0,
            "is_suspicious_domain": 1,
            "domain_risk_tier": "high",
        }

    # Unknown but syntactically present domains are treated as higher-risk in sparse-input mode
    return {
        "normalized": normalized,
        "is_common_domain": 0,
        "is_suspicious_domain": 1,
        "domain_risk_tier": "high",
    }


def _stable_transaction_dt_from_input(transaction: dict) -> float:
    """
    Deterministic pseudo-time for simplified inputs when no explicit time is provided.
    Produces a stable second-of-day value in [0, 86399].
    """
    transaction_hour = transaction.get("transaction_hour")
    if transaction_hour not in (None, ""):
        try:
            hour = int(transaction_hour)
            if 0 <= hour <= 23:
                return float(hour * 3600)
        except Exception:
            pass

    seed_parts = [
        str(transaction.get("amount", "")),
        str(transaction.get("payment_method", "")),
        str(transaction.get("card_type", "")),
        str(transaction.get("card_network", "")),
        str(transaction.get("billing_country", "")),
        str(transaction.get("shipping_country", "")),
        str(transaction.get("email_domain", "")),
        str(transaction.get("receiver_email_domain", "")),
    ]
    seed = "|".join(seed_parts).strip().lower()
    if not seed:
        seed = "default_transaction_seed"

    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    return float(int(digest[:8], 16) % 86400)


def _map_payment_method_to_productcd(value: str | None, amount: float = 0.0) -> str:
    s = str(value).strip().lower() if value is not None else ""

    if s == "card":
        if amount > VERY_HIGH_AMOUNT_THRESHOLD:
            return "W"
        if amount > HIGH_AMOUNT_THRESHOLD:
            return "C"
        return "H"

    if s == "online":
        return "C"

    if s == "transfer":
        return "H"

    return "W"


def _normalize_card_network(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    aliases = {
        "mc": "mastercard",
        "master card": "mastercard",
        "americanexpress": "american express",
        "american-express": "american express",
    }
    return aliases.get(s, s)


def _normalize_device_type(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    if s in {"phone", "mobile", "smartphone", "ios", "android"}:
        return "mobile"
    if s in {"desktop", "pc", "laptop", "computer"}:
        return "desktop"
    if s in {"tablet", "ipad"}:
        return "tablet"
    return s


def _safe_float(value, default=np.nan):
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def extract_inference_risk_signals(raw_row: dict, feature_pipeline: dict) -> dict:
    amount = _safe_float(raw_row.get("TransactionAmt"), 0.0)
    txdt = _safe_float(raw_row.get("TransactionDT"), 0.0)

    payer_domain_info = classify_email_domain(raw_row.get("P_emaildomain"))
    receiver_domain_info = classify_email_domain(raw_row.get("R_emaildomain"))

    payer_domain = payer_domain_info["normalized"]
    receiver_domain = receiver_domain_info["normalized"]

    card_network = _normalize_card_network(raw_row.get("card4"))
    device_type = _normalize_device_type(raw_row.get("DeviceType"))

    aggregate_maps = feature_pipeline.get("aggregate_maps", {}) or {}
    addr_map = aggregate_maps.get("addr1_transaction_mean", {}) or {}
    card_map = aggregate_maps.get("card1_transaction_mean", {}) or {}
    global_mean = float(aggregate_maps.get("global_transaction_mean", 100.0) or 100.0)
    std_proxy = max(global_mean * 2.0, 1.0)

    addr1 = raw_row.get("addr1")
    card1 = raw_row.get("card1")

    missing_identity_count = 0
    for k in ["card1", "card4", "addr1", "P_emaildomain", "DeviceType"]:
        v = raw_row.get(k)
        if v in (None, "", np.nan):
            missing_identity_count += 1

    suspicious_domain_flag = int(
        payer_domain_info["is_suspicious_domain"] or receiver_domain_info["is_suspicious_domain"]
    )
    email_domain_mismatch_flag = int(
        payer_domain is not None and receiver_domain is not None and payer_domain != receiver_domain
    )
    rare_card_flag = int(card_network is not None and card_network not in COMMON_CARD_NETWORKS)
    unusual_addr_flag = int(addr1 not in (None, "", np.nan) and str(addr1) not in {str(k) for k in addr_map.keys()})
    missing_identity_signal = int(missing_identity_count >= 3)

    high_amount_flag = int(amount >= HIGH_AMOUNT_THRESHOLD)
    very_high_amount_flag = int(amount >= VERY_HIGH_AMOUNT_THRESHOLD)
    extreme_amount_flag = int(amount >= EXTREME_AMOUNT_THRESHOLD)

    amount_zstyle_proxy = max(0.0, (amount - global_mean) / std_proxy)
    amount_bin = (
        "extreme" if amount >= EXTREME_AMOUNT_THRESHOLD else
        "very_high" if amount >= VERY_HIGH_AMOUNT_THRESHOLD else
        "high" if amount >= HIGH_AMOUNT_THRESHOLD else
        "normal"
    )
    night_transaction_flag = int(((txdt // 3600) % 24) in [0, 1, 2, 3, 4, 5])

    known_card = card1 not in (None, "", np.nan) and str(card1) in {str(k) for k in card_map.keys()}
    rare_card_flag = max(rare_card_flag, int(card1 not in (None, "", np.nan) and not known_card))

    return {
        "high_amount_flag": high_amount_flag,
        "very_high_amount_flag": very_high_amount_flag,
        "extreme_amount_flag": extreme_amount_flag,
        "email_domain_mismatch_flag": email_domain_mismatch_flag,
        "suspicious_domain_flag": suspicious_domain_flag,
        "rare_card_flag": rare_card_flag,
        "unusual_addr_flag": unusual_addr_flag,
        "missing_identity_signal": missing_identity_signal,
        "night_transaction_flag": night_transaction_flag,
        "amount_bin": amount_bin,
        "amount_zstyle_proxy": round(float(amount_zstyle_proxy), 4),
        "normalized_payer_domain": payer_domain,
        "normalized_receiver_domain": receiver_domain,
        "payer_domain_risk_tier": payer_domain_info["domain_risk_tier"],
        "receiver_domain_risk_tier": receiver_domain_info["domain_risk_tier"],
        "is_common_payer_domain": payer_domain_info["is_common_domain"],
        "is_common_receiver_domain": receiver_domain_info["is_common_domain"],
        "is_suspicious_payer_domain": payer_domain_info["is_suspicious_domain"],
        "is_suspicious_receiver_domain": receiver_domain_info["is_suspicious_domain"],
        "normalized_card_network": card_network,
        "normalized_device_type": device_type,
    }

def adjust_probability_with_signals(base_probability: float, signals: dict) -> dict:
    base_probability = float(np.clip(base_probability, 0.0, 1.0))
    boost = 0.0

    for key, weight in RISK_ADJUSTMENT_WEIGHTS.items():
        if key in {"amount_zstyle_proxy", "multi_signal_bonus", "extreme_combo_bonus"}:
            continue
        boost += float(signals.get(key, 0)) * float(weight)

    if float(signals.get("amount_zstyle_proxy", 0.0)) >= 3.0:
        boost += RISK_ADJUSTMENT_WEIGHTS["amount_zstyle_proxy"]
    if float(signals.get("amount_zstyle_proxy", 0.0)) >= 8.0:
        boost += RISK_ADJUSTMENT_WEIGHTS["amount_zstyle_proxy"]

    signal_count = sum(
        int(bool(signals.get(k, 0)))
        for k in [
            "high_amount_flag", "very_high_amount_flag", "extreme_amount_flag",
            "email_domain_mismatch_flag", "suspicious_domain_flag",
            "rare_card_flag", "unusual_addr_flag", "missing_identity_signal",
            "night_transaction_flag",
        ]
    )
    if signal_count >= 3:
        boost += RISK_ADJUSTMENT_WEIGHTS["multi_signal_bonus"]

    if signals.get("very_high_amount_flag", 0) and (
        signals.get("suspicious_domain_flag", 0) or signals.get("email_domain_mismatch_flag", 0)
    ):
        boost += RISK_ADJUSTMENT_WEIGHTS["extreme_combo_bonus"]

    if signals.get("extreme_amount_flag", 0) and signals.get("missing_identity_signal", 0):
        boost += RISK_ADJUSTMENT_WEIGHTS["extreme_combo_bonus"]

    adjusted_probability = base_probability + (1.0 - base_probability) * boost
    adjusted_probability = float(np.clip(adjusted_probability, base_probability, 0.995))

    # =========================
    # PATCH: Binary Verdict
    # =========================

    FRAUD_THRESHOLD = 0.60  # You can tune this

    if adjusted_probability >= FRAUD_THRESHOLD:
        final_verdict = "FRAUD"
        if adjusted_probability >= 0.85:
            adjusted_risk_level = "CRITICAL"
        else:
            adjusted_risk_level = "HIGH"
    else:
        final_verdict = "LEGITIMATE"
        if adjusted_probability >= 0.30:
            adjusted_risk_level = "MEDIUM"
        else:
            adjusted_risk_level = "LOW"

    return {
        "base_probability": round(base_probability, 6),
        "adjusted_risk_score": round(adjusted_probability, 6),
        "adjusted_risk_level": adjusted_risk_level,
        "final_verdict": final_verdict,
        "signal_count": signal_count,
        "applied_signals": signals,
    }

def preprocess_for_inference(raw_input_df: pd.DataFrame, feature_pipeline: dict) -> pd.DataFrame:
    df = raw_input_df.copy()

    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    raw_columns = feature_pipeline.get("raw_columns_before_cleaning", [])
    for col in raw_columns:
        if col not in df.columns:
            df[col] = np.nan

    df = df[[c for c in raw_columns if c in df.columns]].copy() if raw_columns else df

    columns_to_drop = feature_pipeline.get("columns_to_drop", [])
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], errors="ignore")

    numeric_fill = feature_pipeline.get("numeric_fill_values", {})
    categorical_fill = feature_pipeline.get("categorical_fill_values", {})

    for col, value in numeric_fill.items():
        if col not in df.columns:
            df[col] = value
    for col, value in categorical_fill.items():
        if col not in df.columns:
            df[col] = value

    numeric_cols = list(numeric_fill.keys())
    categorical_cols = list(categorical_fill.keys())

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(numeric_fill[col])
    for col in categorical_cols:
        df[col] = df[col].fillna(categorical_fill[col]).astype(str)

    if "P_emaildomain" in df.columns:
        df["P_emaildomain"] = df["P_emaildomain"].apply(normalize_email_domain).fillna(categorical_fill.get("P_emaildomain", "Unknown"))
    if "R_emaildomain" in df.columns:
        df["R_emaildomain"] = df["R_emaildomain"].apply(normalize_email_domain).fillna(categorical_fill.get("R_emaildomain", "Unknown"))
    if "card4" in df.columns:
        df["card4"] = df["card4"].apply(_normalize_card_network).fillna(categorical_fill.get("card4", "Unknown"))
    if "DeviceType" in df.columns:
        df["DeviceType"] = df["DeviceType"].apply(_normalize_device_type).fillna(categorical_fill.get("DeviceType", "Unknown"))

    aggregate_maps = feature_pipeline.get("aggregate_maps", {}) or {}
    card_mean_map = aggregate_maps.get("card1_transaction_mean", {}) or {}
    card1_count_map = aggregate_maps.get("card1_count_map", {}) or {}
    addr_mean_map = aggregate_maps.get("addr1_transaction_mean", {}) or {}
    global_mean = float(aggregate_maps.get("global_transaction_mean", 0.0) or 0.0)

    if "TransactionAmt" in df.columns:
        amt = pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(0)
        df["TransactionAmt_log"] = np.log1p(amt)

    if "TransactionDT" in df.columns:
        txdt = pd.to_numeric(df["TransactionDT"], errors="coerce").fillna(0)
        df["transaction_hour"] = ((txdt // 3600) % 24).astype(float)
        df["transaction_day"] = ((txdt // 86400) % 7).astype(float)
        df["is_night_transaction"] = ((df["transaction_hour"] >= 0) & (df["transaction_hour"] <= 5)).astype(int)

    if "TransactionAmt" in df.columns and "card1" in df.columns:
        card_key = df["card1"].astype(str)
        card_mean_series = card_key.map(card_mean_map).fillna(global_mean if global_mean > 0 else 1.0).astype(float)
        df["TransactionAmt_to_card_mean"] = pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(0) / (card_mean_series + 1.0)
        df["card1_count"] = card_key.map(card1_count_map).fillna(1).astype(float)

    if "TransactionAmt" in df.columns and "addr1" in df.columns:
        addr_key = df["addr1"].astype(str)
        addr_mean_series = addr_key.map(addr_mean_map).fillna(global_mean if global_mean > 0 else 1.0).astype(float)
        df["amt_addr_ratio"] = pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(0) / (addr_mean_series + 1.0)

    df = _apply_saved_label_encoding(df, feature_pipeline.get("encoder_classes", {}))

    feature_names = feature_pipeline.get("feature_names", [])
    defaults = feature_pipeline.get("feature_default_values", {})
    for col in feature_names:
        if col not in df.columns:
            df[col] = defaults.get(col, 0)

    X = df[feature_names].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X

# ==========================================
#  TOOL 6 — Train XGBoost
# ==========================================
def train_xgboost(X_train, y_train) -> dict:
    """Trains an XGBoost classifier."""
    print("  [MLTool] Training XGBoost model...")

    fraud_count = int((y_train == 1).sum())
    non_fraud   = int((y_train == 0).sum())
    scale_weight = non_fraud / fraud_count if fraud_count > 0 else 1

    model = xgb.XGBClassifier(
        n_estimators      = 200,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_weight,
        use_label_encoder = False,
        eval_metric       = "auc",
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        verbosity         = 0
    )
    model.fit(X_train, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    joblib.dump(model, model_path)

    return {
        "model": model, "model_name": "XGBoost",
        "model_path": model_path,
        "summary": "XGBoost model trained and saved successfully."
    }


# ==========================================
#  TOOL 7 — Train LightGBM
# ==========================================
def train_lightgbm(X_train, y_train) -> dict:
    """Trains a LightGBM classifier."""
    print("  [MLTool] Training LightGBM model...")

    model = lgb.LGBMClassifier(
        n_estimators    = 200,
        max_depth       = 6,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        class_weight    = "balanced",
        random_state    = RANDOM_STATE,
        n_jobs          = -1,
        verbose         = -1
    )
    model.fit(X_train, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "lightgbm_model.pkl")
    joblib.dump(model, model_path)

    return {
        "model": model, "model_name": "LightGBM",
        "model_path": model_path,
        "summary": "LightGBM model trained and saved successfully."
    }


# ==========================================
#  TOOL 8 (NEW) — Train CatBoost
# ==========================================
def train_catboost(X_train, y_train) -> dict:
    """
    Trains a CatBoost classifier.

    WHAT IS CATBOOST?
    CatBoost (Categorical Boosting) by Yandex is a gradient boosting
    library specifically designed to handle categorical features natively
    without explicit encoding. It uses ordered boosting to prevent
    target leakage and is often the most accurate of the three boosters.

    Why add it to the ensemble?
      - XGBoost: best at capturing complex non-linear interactions
      - LightGBM: fastest, great recall
      - CatBoost: best at categorical features, less overfitting
      - Together (stacked): higher AUC than any single model
    """
    print("  [MLTool] Training CatBoost model...")

    fraud_count  = int((y_train == 1).sum())
    non_fraud    = int((y_train == 0).sum())
    scale_weight = non_fraud / fraud_count if fraud_count > 0 else 1

    model = CatBoostClassifier(
        iterations       = 200,
        depth            = 6,
        learning_rate    = 0.05,
        scale_pos_weight = scale_weight,
        random_seed      = RANDOM_STATE,
        verbose          = 0,           # silent training
        eval_metric      = "AUC",
        thread_count     = -1
    )
    model.fit(X_train, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "catboost_model.pkl")
    joblib.dump(model, model_path)

    return {
        "model": model, "model_name": "CatBoost",
        "model_path": model_path,
        "summary": "CatBoost model trained and saved successfully."
    }


# ==========================================
#  TOOL 9 (NEW) — Build Stacking Ensemble with OOF
# ==========================================
def build_stacking_ensemble(
    X_train, y_train,
    xgb_model, lgb_model, cat_model,
    X_test
) -> dict:
    """
    Builds a TRUE stacking ensemble using Out-of-Fold (OOF) predictions.

    WHY OOF? — Preventing Data Leakage
    ====================================
    Naive stacking generates base model predictions on the SAME data
    they were trained on. The meta-learner then learns from predictions
    that are unrealistically good (the base models memorized training data).
    This inflates validation scores and causes overfitting in production.

    OOF stacking solves this:
      1. Split training data into K folds (we use 5)
      2. For each fold:
         - Train base models on the other K-1 folds
         - Predict on the held-out fold (models never saw this fold)
      3. This gives us N training predictions where each row was predicted
         by a model that did NOT train on it — honest predictions!
      4. Train the meta-learner on these OOF predictions
      5. For TEST predictions: average predictions across all K-fold models

    Result: a stacking ensemble that is immune to data leakage and
    generalises correctly to unseen data.

    Architecture:
      Layer 1 (Base):  XGBoost | LightGBM | CatBoost
                            |         |          |
      Layer 2 (Meta):  Logistic Regression on [p_xgb, p_lgb, p_cat]
    """
    print(f"  [Stacking] Building OOF Stacking Ensemble ({N_STACK_FOLDS} folds)...")
    n = len(X_train)
    y_arr = np.array(y_train)

    # Storage for OOF predictions (shape: n_train x 3 base models)
    oof_preds = np.zeros((n, 3))

    # Storage for test predictions from each fold (shape: n_test x K x 3)
    if hasattr(X_test, "values"):
        X_test_arr = X_test.values
    else:
        X_test_arr = np.array(X_test)

    test_preds_folds = np.zeros((len(X_test_arr), N_STACK_FOLDS, 3))

    skf = StratifiedKFold(n_splits=N_STACK_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    if hasattr(X_train, "values"):
        X_train_arr = X_train.values
    else:
        X_train_arr = np.array(X_train)

    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_arr, y_arr)):
        print(f"    Fold {fold_idx+1}/{N_STACK_FOLDS}...")

        X_fold_train = X_train_arr[train_idx]
        y_fold_train = y_arr[train_idx]
        X_fold_val   = X_train_arr[val_idx]

        # Convert back to DataFrames for compatibility
        if feature_names:
            X_fold_train_df = pd.DataFrame(X_fold_train, columns=feature_names)
            X_fold_val_df   = pd.DataFrame(X_fold_val,   columns=feature_names)
            X_test_df       = pd.DataFrame(X_test_arr,   columns=feature_names)
        else:
            X_fold_train_df = X_fold_train
            X_fold_val_df   = X_fold_val
            X_test_df       = X_test_arr

        # --- Clone and fit XGBoost ---
        fraud_count  = int((y_fold_train == 1).sum())
        non_fraud    = int((y_fold_train == 0).sum())
        scale_weight = non_fraud / fraud_count if fraud_count > 0 else 1

        xgb_fold = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_weight, use_label_encoder=False,
            eval_metric="auc", random_state=RANDOM_STATE,
            n_jobs=-1, verbosity=0
        )
        xgb_fold.fit(X_fold_train_df, y_fold_train)
        oof_preds[val_idx, 0]        = xgb_fold.predict_proba(X_fold_val_df)[:, 1]
        test_preds_folds[:, fold_idx, 0] = xgb_fold.predict_proba(X_test_df)[:, 1]

        # --- Clone and fit LightGBM ---
        lgb_fold = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1
        )
        lgb_fold.fit(X_fold_train_df, y_fold_train)
        oof_preds[val_idx, 1]        = lgb_fold.predict_proba(X_fold_val_df)[:, 1]
        test_preds_folds[:, fold_idx, 1] = lgb_fold.predict_proba(X_test_df)[:, 1]

        # --- Clone and fit CatBoost ---
        cat_fold = CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.05,
            scale_pos_weight=scale_weight, random_seed=RANDOM_STATE,
            verbose=0, eval_metric="AUC", thread_count=-1
        )
        cat_fold.fit(X_fold_train_df, y_fold_train)
        oof_preds[val_idx, 2]        = cat_fold.predict_proba(X_fold_val_df)[:, 1]
        test_preds_folds[:, fold_idx, 2] = cat_fold.predict_proba(X_test_df)[:, 1]

    # Average test predictions across folds (one per base model)
    test_preds_avg = test_preds_folds.mean(axis=1)   # shape: (n_test, 3)

    # OOF AUCs for each base model
    oof_auc_xgb = roc_auc_score(y_arr, oof_preds[:, 0])
    oof_auc_lgb = roc_auc_score(y_arr, oof_preds[:, 1])
    oof_auc_cat = roc_auc_score(y_arr, oof_preds[:, 2])

    print(f"    OOF AUC — XGBoost: {oof_auc_xgb:.4f}  "
          f"LightGBM: {oof_auc_lgb:.4f}  CatBoost: {oof_auc_cat:.4f}")

    # --- Train Meta-Learner on OOF predictions ---
    print("  [Stacking] Training meta-learner (Logistic Regression) on OOF predictions...")
    meta_learner = LogisticRegression(
        C=1.0, max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1
    )
    meta_learner.fit(oof_preds, y_arr)

    # Meta-learner OOF AUC
    oof_meta_proba = meta_learner.predict_proba(oof_preds)[:, 1]
    oof_auc_meta   = roc_auc_score(y_arr, oof_meta_proba)
    print(f"    Meta-learner OOF AUC: {oof_auc_meta:.4f}")

    # Save stacking artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(meta_learner, os.path.join(MODELS_DIR, "stacking_meta_learner.pkl"))
    np.save(os.path.join(MODELS_DIR, "oof_preds.npy"),      oof_preds)
    np.save(os.path.join(MODELS_DIR, "test_preds_avg.npy"), test_preds_avg)

    summary = (
        f"Stacking Ensemble Built (OOF — {N_STACK_FOLDS} folds):\n"
        f"  Base Model OOF AUCs:\n"
        f"    XGBoost  : {oof_auc_xgb:.4f}\n"
        f"    LightGBM : {oof_auc_lgb:.4f}\n"
        f"    CatBoost : {oof_auc_cat:.4f}\n"
        f"  Meta-learner: Logistic Regression\n"
        f"  Meta OOF AUC: {oof_auc_meta:.4f}\n"
        f"  Data leakage: NONE (strict OOF separation)\n"
        f"  Artifacts saved to /models/"
    )

    return {
        "meta_learner"   : meta_learner,
        "oof_preds"      : oof_preds,
        "test_preds_avg" : test_preds_avg,
        "oof_auc_xgb"    : oof_auc_xgb,
        "oof_auc_lgb"    : oof_auc_lgb,
        "oof_auc_cat"    : oof_auc_cat,
        "oof_auc_meta"   : oof_auc_meta,
        "summary"        : summary
    }


# ==========================================
#  TOOL 10 — Predict with Stacking Ensemble
# ==========================================
def predict_with_stack(
    meta_learner,
    xgb_model, lgb_model, cat_model,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Generates fraud probability predictions using the stacking ensemble.

    For new/test data, we run each base model individually,
    assemble their predictions into a feature matrix, then pass
    through the meta-learner.

    Returns array of fraud probabilities (shape: n_samples,)
    """
    p_xgb = xgb_model.predict_proba(X)[:, 1]
    p_lgb = lgb_model.predict_proba(X)[:, 1]
    p_cat = cat_model.predict_proba(X)[:, 1]

    meta_features = np.column_stack([p_xgb, p_lgb, p_cat])
    stack_proba   = meta_learner.predict_proba(meta_features)[:, 1]
    return stack_proba


# ==========================================
#  TOOL 11 — Evaluate Model
# ==========================================
def evaluate_model(model, model_name: str, X_test, y_test,
                   is_stack: bool = False, meta_learner=None,
                   xgb_model=None, lgb_model=None, cat_model=None,
                   test_preds_avg=None) -> dict:
    """
    Evaluates model performance. Handles both single models and
    the stacking ensemble (pass is_stack=True for ensemble).
    """
    print(f"  [MLTool] Evaluating {model_name}...")

    if is_stack and meta_learner is not None:
        if test_preds_avg is not None:
            # Use the pre-computed averaged test predictions (most accurate)
            meta_features = test_preds_avg
            y_proba = meta_learner.predict_proba(meta_features)[:, 1]
        else:
            y_proba = predict_with_stack(meta_learner, xgb_model, lgb_model, cat_model, X_test)
    else:
        y_proba = model.predict_proba(X_test)[:, 1]

    y_pred    = (y_proba >= 0.5).astype(int)
    auc       = roc_auc_score(y_test, y_proba)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    summary = (
        f"Evaluation Results — {model_name}:\n"
        f"  AUC-ROC   : {auc:.4f}  {'[OK] GOOD' if auc >= 0.85 else '⚠ NEEDS IMPROVEMENT'}\n"
        f"  F1 Score  : {f1:.4f}\n"
        f"  Precision : {precision:.4f}\n"
        f"  Recall    : {recall:.4f}\n"
        f"  ----------------------\n"
        f"  Confusion Matrix:\n"
        f"    True Negatives  (correct non-fraud) : {tn:,}\n"
        f"    False Positives (wrong fraud alarm)  : {fp:,}\n"
        f"    False Negatives (missed real fraud)  : {fn:,}\n"
        f"    True Positives  (caught fraud)       : {tp:,}\n"
    )

    return {
        "model_name": model_name, "auc": auc, "f1": f1,
        "precision": precision, "recall": recall,
        "y_proba": y_proba, "confusion_matrix": cm,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "summary": summary
    }


# ==========================================
#  TOOL 12 — Predict on Single Transaction
# ==========================================
def predict_transaction(model, transaction: dict, feature_names: list,
                        is_stack: bool = False, meta_learner=None,
                        xgb_model=None, lgb_model=None, cat_model=None) -> dict:
    """
    Predicts fraud probability for a single transaction.
    Supports both single-model and stacking ensemble modes.
    """
    row = pd.DataFrame([transaction])
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_names]

    if is_stack and meta_learner is not None:
        fraud_prob = float(predict_with_stack(meta_learner, xgb_model, lgb_model, cat_model, row)[0])
    else:
        fraud_prob = float(model.predict_proba(row)[0][1])

    if fraud_prob < 0.3:
        risk_level = "LOW"
    elif fraud_prob < 0.5:
        risk_level = "MEDIUM"
    elif fraud_prob < 0.75:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return {
        "fraud_probability": round(fraud_prob, 4),
        "is_fraud"         : fraud_prob >= 0.5,
        "risk_level"       : risk_level
    }



def compute_feature_hash(feature_names: list) -> str:
    return hashlib.sha256("|".join(feature_names).encode("utf-8")).hexdigest() if feature_names else ""


def validate_required_artifacts(models_dir: str = MODELS_DIR, feature_pipeline_path: str = FEATURE_PIPELINE_PATH,
                                metadata_path: str = TRAINING_METADATA_PATH) -> dict:
    missing = []
    paths = {}
    for name in REQUIRED_TRAINED_ARTIFACTS:
        path = feature_pipeline_path if name == "feature_pipeline.pkl" else os.path.join(models_dir, name)
        paths[name] = path
        if not os.path.exists(path):
            missing.append(name)
    metadata_exists = os.path.exists(metadata_path)
    if missing:
        raise FileNotFoundError(
            "Missing trained artifacts: " + ", ".join(missing) + ". Run ⁠ python main.py --pipeline ⁠ first."
        )
    return {"paths": paths, "metadata_exists": metadata_exists, "metadata_path": metadata_path}


def validate_artifact_compatibility(feature_pipeline: dict, models: dict, metadata: dict | None = None) -> None:
    feature_names = feature_pipeline.get("feature_names", [])
    if not feature_names:
        raise ValueError("feature_pipeline.pkl is missing feature_names. Retrain pipeline.")
    expected_count = len(feature_names)
    expected_hash = compute_feature_hash(feature_names)

    for model_name, model in models.items():
        n_features = getattr(model, "n_features_in_", None)

        # Some libraries (notably CatBoost) may expose 0 or None here.
        # Treat that as "unknown" and skip strict validation.
        if n_features in (None, 0):
            continue

        # Base learners must match raw feature count
        if model_name in {"xgb_model", "lgb_model", "cat_model"}:
            if int(n_features) != expected_count:
                raise ValueError("Artifacts inconsistent, retrain pipeline")

        # Meta learner and stacked best model operate on 3 base-model outputs
        elif model_name in {"meta_learner", "best_model"}:
            if int(n_features) not in (3, expected_count):
                raise ValueError("Artifacts inconsistent, retrain pipeline")

    if metadata:
        metadata_count = metadata.get("feature_count")
        metadata_hash = metadata.get("feature_hash")
        if metadata_count not in (None, expected_count):
            raise ValueError("Artifacts inconsistent, retrain pipeline")
        if metadata_hash not in (None, expected_hash):
            raise ValueError("Artifacts inconsistent, retrain pipeline")


def _canonicalize_input_keys(transaction: dict) -> tuple[dict, list[str]]:
    schema = SINGLE_TRANSACTION_INPUT_SCHEMA
    aliases = {str(k).lower(): v for k, v in schema.get("aliases", {}).items()}
    canonical = {}
    unknown = []
    for key, value in transaction.items():
        if key in aliases.values() or key in schema.get("required", []) or key in schema.get("optional", []):
            canonical[key] = value
            continue
        mapped = aliases.get(str(key).strip().lower())
        if mapped:
            canonical[mapped] = value
        else:
            unknown.append(str(key))
    return canonical, unknown

def _stable_country_code(value: str | None, default: int = 0) -> int:
    """
    Deterministically maps a country string to a numeric code so it can be placed
    into existing addr1/addr2 fields without retraining the model.
    """
    if value is None:
        return default
    s = str(value).strip().lower()
    if not s or s in {"unknown", "na", "n/a", "none", "null"}:
        return default
    digest = hashlib.md5(s.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 997) + 1  # stable 1..997


def _map_payment_method_to_productcd(value: str | None, amount: float = 0.0) -> str:
    s = str(value).strip().lower() if value is not None else ""

    if s == "card":
        if amount > VERY_HIGH_AMOUNT_THRESHOLD:
            return "W"
        if amount > HIGH_AMOUNT_THRESHOLD:
            return "C"
        return "H"

    if s == "online":
        return "C"

    if s == "transfer":
        return "H"

    return "W"


def _normalize_card_type(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"credit", "debit"}:
        return s
    return None


def _normalize_card_network_user(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    mapping = {
        "visa": "visa",
        "mastercard": "mastercard",
        "master card": "mastercard",
        "mc": "mastercard",
        "amex": "american express",
        "american express": "american express",
        "discover": "discover",
        "unknown": "unknown",
    }
    return mapping.get(s, "unknown")


def _is_simplified_schema(transaction: dict) -> bool:
    simplified_keys = {
        "amount",
        "payment_method",
        "card_type",
        "card_network",
        "billing_country",
        "shipping_country",
        "email_domain",
        "receiver_email_domain",
    }
    return any(k in transaction for k in simplified_keys)


def _is_legacy_schema(transaction: dict) -> bool:
    legacy_keys = {
        "TransactionAmt",
        "ProductCD",
        "card4",
        "card6",
        "addr1",
        "addr2",
        "P_emaildomain",
        "R_emaildomain",
    }
    return any(k in transaction for k in legacy_keys)


def _validate_simplified_transaction_input(transaction: dict) -> list[str]:
    errors = []
    schema = SIMPLIFIED_TRANSACTION_INPUT_SCHEMA

    if "amount" not in transaction or transaction.get("amount") in (None, ""):
        errors.append(
            "Missing required field 'amount'. Example: {\"amount\": 149.99, \"card_network\": \"visa\"}"
        )
        return errors

    try:
        amount = float(transaction.get("amount"))
        if amount < 0:
            errors.append("Field 'amount' must be a non-negative number.")
    except Exception:
        errors.append("Field 'amount' must be numeric. Example: 149.99")

    for field in ["payment_method", "card_type", "card_network"]:
        value = transaction.get(field)
        if value in (None, ""):
            continue
        allowed = schema["allowed_values"].get(field, set())
        if str(value).strip().lower() not in allowed:
            errors.append(
                f"Invalid '{field}'. Allowed values: {', '.join(sorted(allowed))}."
            )

    hour_value = transaction.get("transaction_hour")
    if hour_value not in (None, ""):
        try:
            hour_int = int(hour_value)
            if hour_int < 0 or hour_int > 23:
                errors.append("Invalid 'transaction_hour'. Allowed range: 0 to 23.")
        except Exception:
            errors.append("Field 'transaction_hour' must be an integer between 0 and 23.")

    return errors

def _map_simplified_to_internal_ieee(transaction: dict) -> dict:
    """
    Maps the simple user-facing schema into the existing internal IEEE-style
    transaction keys expected by the current inference pipeline.
    """
    amount = float(transaction.get("amount", 0) or 0)

    payer_domain_info = classify_email_domain(transaction.get("email_domain"))
    receiver_domain_info = classify_email_domain(transaction.get("receiver_email_domain"))

    payer_domain = payer_domain_info["normalized"]
    receiver_domain = receiver_domain_info["normalized"]
    if receiver_domain is None and payer_domain is not None:
        receiver_domain = payer_domain

    transaction_dt = _stable_transaction_dt_from_input(transaction)

    internal = {
        "TransactionAmt": amount,
        "TransactionDT": transaction_dt,
        "ProductCD": _map_payment_method_to_productcd(
            transaction.get("payment_method"),
            amount=amount,
        ),
        "card4": _normalize_card_network_user(transaction.get("card_network")),
        "card6": _normalize_card_type(transaction.get("card_type")),
        "addr1": _stable_country_code(transaction.get("billing_country"), default=0),
        "addr2": _stable_country_code(transaction.get("shipping_country"), default=0),
        "P_emaildomain": payer_domain,
        "R_emaildomain": receiver_domain,
        "dist1": np.nan,
        "DeviceType": None,
        "DeviceInfo": None,
    }

    if internal["ProductCD"] is None:
        internal["ProductCD"] = "W"

    if internal["card4"] is None:
        internal["card4"] = "unknown"

    return internal

def map_basic_transaction_input(transaction: dict) -> pd.DataFrame:
    """
    Accepts:
    1) New simplified end-user schema
    2) Legacy IEEE-style schema for backward compatibility

    Returns a one-row DataFrame using the existing internal IEEE-style field names.
    """
    if not isinstance(transaction, dict):
        raise TypeError("Transaction input must be a JSON object / Python dict.")
    if not transaction:
        raise ValueError("Transaction input is empty. Please provide transaction details.")

    # -----------------------------
    # Path A: simplified user schema
    # -----------------------------
    if _is_simplified_schema(transaction):
        errors = _validate_simplified_transaction_input(transaction)
        if errors:
            raise ValueError("Invalid transaction input: " + " ".join(errors))

        mapped = _map_simplified_to_internal_ieee(transaction)
        return pd.DataFrame([mapped])

    # -----------------------------
    # Path B: legacy schema support
    # -----------------------------
    if _is_legacy_schema(transaction):
        schema = SINGLE_TRANSACTION_INPUT_SCHEMA
        canonical, unknown = _canonicalize_input_keys(transaction)

        missing_required = [
            field for field in schema.get("required", [])
            if field not in canonical or canonical.get(field) in (None, "")
        ]
        if missing_required:
            accepted = sorted(
                set(schema.get("required", []) + list(schema.get("aliases", {}).keys()))
            )
            raise ValueError(
                schema.get("required_message", "Missing required transaction fields.")
                + f" Missing: {', '.join(missing_required)}. "
                f"Accepted keys include: {', '.join(accepted[:20])}."
            )

        numeric_fields = {
            "TransactionAmt", "TransactionDT", "card1", "card2", "card3",
            "card5", "addr1", "addr2", "dist1"
        }
        normalized = {}
        errors = []

        for key, value in canonical.items():
            if key in numeric_fields:
                parsed = _safe_float(value, np.nan)
                if pd.isna(parsed) and value not in (None, ""):
                    errors.append(f"Field '{key}' must be numeric, got {value!r}.")
                else:
                    normalized[key] = parsed
            else:
                normalized[key] = None if value in (None, "") else str(value).strip()

        if errors:
            raise ValueError("Invalid legacy transaction input: " + " ".join(errors))

        amt = float(normalized.get("TransactionAmt", 0) or 0)
        if amt < 0:
            raise ValueError("Invalid legacy transaction input: TransactionAmt must be non-negative.")

        normalized["P_emaildomain"] = normalize_email_domain(normalized.get("P_emaildomain"))
        normalized["R_emaildomain"] = normalize_email_domain(normalized.get("R_emaildomain"))
        normalized["card4"] = _normalize_card_network(normalized.get("card4"))
        normalized["DeviceType"] = _normalize_device_type(normalized.get("DeviceType"))

        if normalized.get("R_emaildomain") is None and normalized.get("P_emaildomain") is not None:
            normalized["R_emaildomain"] = normalized["P_emaildomain"]

        if normalized.get("TransactionDT") is None or pd.isna(normalized.get("TransactionDT")):
            normalized["TransactionDT"] = 0.0

        if normalized.get("ProductCD") is None:
            if amt >= VERY_HIGH_AMOUNT_THRESHOLD:
                normalized["ProductCD"] = "W"
            elif amt >= HIGH_AMOUNT_THRESHOLD:
                normalized["ProductCD"] = "C"
            else:
                normalized["ProductCD"] = "H"

        if normalized.get("card6") is None and normalized.get("card4") in {"visa", "mastercard", "discover"}:
            normalized["card6"] = "debit"

        if normalized.get("DeviceType") is None:
            normalized["DeviceType"] = "unknown"

        cleaned = {k: v for k, v in normalized.items() if not str(k).startswith("_")}
        return pd.DataFrame([cleaned])

    # -----------------------------
    # Neither schema matched
    # -----------------------------
    raise ValueError(
        "Unrecognized transaction format. "
        "Provide either the simplified schema "
        "{amount, payment_method, card_type, card_network, billing_country, shipping_country, "
        "email_domain, receiver_email_domain} "
        "or the legacy internal schema using fields like TransactionAmt and card4."
    )

def filter_user_visible_context(raw_row: dict, signals: dict) -> dict:
    """
    Keeps only user-provided fields and explicitly derived signals.
    Removes any hidden/model-only/internal features.
    """

    visible_fields = {}

    # Only include fields actually provided
    for k, v in raw_row.items():
        if v not in (None, "", "nan"):
            visible_fields[k] = v

    # Only include ACTIVE signals (value = 1 or meaningful)
    active_signals = {}
    for k, v in signals.items():
        if isinstance(v, (int, float)) and v:
            active_signals[k] = v
        elif isinstance(v, str) and v not in ("normal", "", None):
            active_signals[k] = v

    return {
        "fields": visible_fields,
        "signals": active_signals
    }