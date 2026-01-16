# HDFS paths for source data and final outputs
DATA_PATH = "/user/s2578018/all_reviews.csv.gz"
METRICS_CSV_PATH = "steam_metrics_results"
REPORT_PATH = "steam_summary_report"
# Minimum review count 
MIN_REVIEWS_THRESHOLD = 30

# Split ratio: 80/20 train test
TRAIN_TEST_SPLIT = [0.8, 0.2]

# K-Fold Cross-Validation folds 
NUM_FOLDS = 3

# Grids of hyperparameters to test during training
PARAM_GRIDS = {
    "LinearRegression": {
        # Regularization parameters
        "regParam": [0.1, 0.01, 0.001],
        # Penalty mixing parameter
        "elasticNetParam": [0.0, 0.5, 1.0]
    },
    "RandomForest": {
        "numTrees": [20, 50, 100],
        "maxDepth": [5, 10]
    },
    "XGBoost": {
        "max_depth": [3, 6, 9], 
        "learning_rate": [0.1, 0.01],
        "n_estimators": [100, 200]
    }
}