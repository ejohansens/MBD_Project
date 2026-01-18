# HDFS paths for source data and final outputs
DATA_PATH = "/user/s2578018/all_reviews.csv.gz"
METRICS_CSV_PATH = "steam_metrics_results"
REPORT_PATH = "steam_summary_report"

# Minimum review count 
MIN_REVIEWS_THRESHOLD = 30

# Split ratio: 80/20 train test
TRAIN_TEST_SPLIT = [0.8, 0.2]

# --- These are now optimized for a Simple Fit ---

# Single "Best" Parameters for the models
MODEL_PARAMS = {
    "LinearRegression": {
        "regParam": 0.01,
        "elasticNetParam": 0.5
    },
    "RandomForest": {
        "numTrees": 100,
        "maxDepth": 10
    },
    "XGBoost": {
        "max_depth": 6, 
        "learning_rate": 0.1,
        "n_estimators": 200
    }
}