from pyspark.sql import SparkSession
from data_processor import SteamDataProcessor
from model_factory import ModelFactory
from results_manager import ResultsManager
from pyspark.ml.evaluation import RegressionEvaluator
import config

def main():
    spark = SparkSession.builder.appName("SteamAnalysis").getOrCreate()
    
    # Load and clean data
    processor = SteamDataProcessor(spark, config.DATA_PATH, config.MIN_REVIEWS_THRESHOLD)
    raw_data = processor.load_clean_data()
    
    # Engineer features and predictors
    ml_ready_data = processor.get_features_and_label(raw_data)
    train, test = ml_ready_data.randomSplit(config.TRAIN_TEST_SPLIT, seed=42)
    
    # Train the models
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
    factory = ModelFactory(train, evaluator)
    
    best_results = factory.run_comparison([
        ("LinearRegression", factory.build_lr()),
        ("RandomForest", factory.build_rf()),
        ("XGBoost", factory.build_xgboost())
    ])
    
    # Evaluate and collect metrics
    summary_metrics = []
    for name, model in best_results.items():
        preds = model.transform(test)
        r2 = evaluator.setMetricName("r2").evaluate(preds)
        rmse = evaluator.setMetricName("rmse").evaluate(preds)
        summary_metrics.append((name, r2, rmse))
    
    # Save
    manager = ResultsManager(spark)
    manager.save_metrics_to_csv(summary_metrics, config.METRICS_CSV_PATH)
    print(manager.save_summary_report(summary_metrics, ml_ready_data.count(), config.REPORT_PATH))

if __name__ == "__main__":
    main()