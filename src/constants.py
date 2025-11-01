ARTIFACT_DIR = Path("./artifacts")

RANDOM_STATE = 42
TEST_SIZE = 0.20

EMAIL_RE = re.compile(r"([A-Za-z0-9._%+\-]+)@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


# TF-IDF knobs (kept lean to avoid timeouts on large data)
TFIDF_MAX_FEATURES = 10000
TFIDF_MIN_DF = 2
TFIDF_NGRAM_RANGE = (1, 1)  # unigrams only for speed/memory

# RandomForest knobs
RF_TREES = 200  # small-ish to keep runtime tame

# Keyword flags (quick heuristics)
KEYWORDS = [
    "viagra",
    "winner",
    "lottery",
    "free",
    "sex",
    "urgent",
    "account",
    "password",
    "bank",
    "verify",
    "click",
    "limited",
    "offer",
    "money",
    "prize",
    "deal",
    "cheap",
]
