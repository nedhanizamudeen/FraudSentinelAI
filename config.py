# FraudSentinel AI — config.py
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "paste_your_api_key")
LLM_MODEL = "llama-3.3-70b-versatile"

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
TRANSACTION_FILE = os.path.join(DATA_DIR, "train_transaction.csv")
IDENTITY_FILE = os.path.join(DATA_DIR, "train_identity.csv")
TEST_TRANSACTION = os.path.join(DATA_DIR, "test_transaction.csv")
TEST_IDENTITY = os.path.join(DATA_DIR, "test_identity.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FEATURE_PIPELINE_PATH = os.path.join(MODELS_DIR, "feature_pipeline.pkl")
TRAINING_METADATA_PATH = os.path.join(MODELS_DIR, "training_metadata.json")
INFERENCE_OUTPUT_DIR = os.path.join(BASE_DIR, "inference_outputs")

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "isFraud"
MIN_AUC_THRESHOLD = 0.85
FRAUD_THRESHOLD = 0.5
MAX_MISSING_RATIO = 0.5
TARGET_CLASS_RATIO = 10
N_STACK_FOLDS = 5
META_LEARNER = "logistic"
N_COUNTERFACTUAL_STEPS = 5
N_COUNTERFACTUAL_TRANSACTIONS = 10
PERTURBATION_STEP = 0.1
PROJECT_NAME = "FraudSentinel AI"
PROJECT_VERSION = "2.3.0-backend"

REQUIRED_TRAINED_ARTIFACTS = [
    "stacking_meta_learner.pkl",
    "xgboost_model.pkl",
    "lightgbm_model.pkl",
    "catboost_model.pkl",
    "feature_pipeline.pkl",
]

SINGLE_TRANSACTION_INPUT_SCHEMA = {
    "required": ["TransactionAmt"],
    "required_message": (
        "Provide at least the transaction amount. "
        "Accepted keys include TransactionAmt, amount, transaction_amount, amt."
    ),
    "optional": [
        "TransactionDT", "ProductCD",
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "dist1",
        "P_emaildomain", "R_emaildomain",
        "DeviceType", "DeviceInfo",
    ],
    "aliases": {
        "amount": "TransactionAmt",
        "transaction_amount": "TransactionAmt",
        "amt": "TransactionAmt",
        "transaction_time": "TransactionDT",
        "timestamp": "TransactionDT",
        "seconds_from_reference": "TransactionDT",
        "product": "ProductCD",
        "product_code": "ProductCD",
        "card_number_prefix": "card1",
        "card_bin": "card1",
        "issuer_bin": "card1",
        "card_network": "card4",
        "network": "card4",
        "card_type": "card6",
        "billing_region": "addr1",
        "billing_zip_region": "addr1",
        "shipping_region": "addr2",
        "distance": "dist1",
        "distance_from_billing": "dist1",
        "payer_email_domain": "P_emaildomain",
        "buyer_email_domain": "P_emaildomain",
        "sender_email_domain": "P_emaildomain",
        "recipient_email_domain": "R_emaildomain",
        "merchant_email_domain": "R_emaildomain",
        "receiver_email_domain": "R_emaildomain",
        "device_type": "DeviceType",
        "device_info": "DeviceInfo",
    },
    "critical_csv_any_of": ["TransactionAmt"],
    "description": "Simplified user fields are normalized into the model input schema; missing optional fields are filled safely.",
}

# --- NEW: simplified user-facing input schema ---
SIMPLIFIED_TRANSACTION_INPUT_SCHEMA = {
    "required": ["amount"],
    "optional": [
        "payment_method",
        "card_type",
        "card_network",
        "billing_country",
        "shipping_country",
        "email_domain",
        "receiver_email_domain",
        "transaction_hour",
    ],
    "allowed_values": {
        "payment_method": {"card", "online", "transfer"},
        "card_type": {"credit", "debit"},
        "card_network": {"visa", "mastercard", "amex", "discover", "unknown"},
    },
    "description": (
        "End users should provide simple transaction fields. "
        "These are mapped internally to the existing IEEE-style inference schema."
    ),
}

# Keep backward compatibility with the older internal-style input schema.
LEGACY_TRANSACTION_INPUT_SCHEMA = {
    "required": ["TransactionAmt"],
    "optional": [
        "TransactionDT", "ProductCD",
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "dist1",
        "P_emaildomain", "R_emaildomain",
        "DeviceType", "DeviceInfo",
    ],
}

COMMON_CARD_NETWORKS = {
    "visa", "mastercard", "discover", "american express", "amex"
}

# Common consumer domains used as lower-risk baseline references
COMMON_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "icloud.com",
    "live.com",
    "msn.com",
    "aol.com",
}

# Higher-risk or less typical domains for simplified-input inference
SUSPICIOUS_EMAIL_DOMAINS = {
    "mailinator.com",
    "guerrillamail.com",
    "10minutemail.com",
    "temp-mail.org",
    "tempmail.com",
    "trashmail.com",
    "sharklasers.com",
    "yopmail.com",
    "dispostable.com",
    "fakeinbox.com",
    "getnada.com",
    "mintemail.com",
    "protonmail.com",
    "mail.ru",
    "yandex.ru",
    "fraud.com",
    "unknown",
}

SUSPICIOUS_EMAIL_DOMAINS = {
    "mailinator.com", "guerrillamail.com", "10minutemail.com", "temp-mail.org",
    "tempmail.com", "trashmail.com", "sharklasers.com", "yopmail.com",
    "dispostable.com", "fakeinbox.com", "getnada.com", "mintemail.com"
}

HIGH_AMOUNT_THRESHOLD = 2000.0
VERY_HIGH_AMOUNT_THRESHOLD = 10000.0
EXTREME_AMOUNT_THRESHOLD = 50000.0

RISK_ADJUSTMENT_WEIGHTS = {
    "high_amount_flag": 0.08,
    "very_high_amount_flag": 0.18,
    "extreme_amount_flag": 0.25,
    "email_domain_mismatch_flag": 0.12,
    "suspicious_domain_flag": 0.10,
    "rare_card_flag": 0.08,
    "unusual_addr_flag": 0.07,
    "missing_identity_signal": 0.10,
    "night_transaction_flag": 0.04,
    "amount_zstyle_proxy": 0.06,
    "multi_signal_bonus": 0.08,
    "extreme_combo_bonus": 0.12,
}