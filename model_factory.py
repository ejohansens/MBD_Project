from pyspark.ml.regression import LinearRegression, RandomForestRegressor

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
import config

# Factory class to build and compare models (LR, RF, XGBoost)
class ModelFactory:
    def __init__(self, train_data, evaluator):
        self.train_data = train_data
        self.evaluator = evaluator

    def build_lr(self):
        # Linear Regression with elastic net penlties
        lr = LinearRegression(featuresCol="features", labelCol="label", weightCol="m1_count")
        grid = ParamGridBuilder() \
               .addGrid(lr.regParam, config.PARAM_GRIDS["LinearRegression"]["regParam"]) \
               .addGrid(lr.elasticNetParam, config.PARAM_GRIDS["LinearRegression"]["elasticNetParam"]) \
               .build()
        return lr, grid

    def build_rf(self):
        # Random Forest Regressor
        rf = RandomForestRegressor(featuresCol="features", labelCol="label")
        grid = ParamGridBuilder() \
               .addGrid(rf.numTrees, config.PARAM_GRIDS["RandomForest"]["numTrees"]) \
               .addGrid(rf.maxDepth, config.PARAM_GRIDS["RandomForest"]["maxDepth"]) \
               .build()
        return rf, grid

    def build_xgboost(self):
            # XGBOOST Regressor
            from xgboost.spark import SparkXGBRegressor
            
            xgb = SparkXGBRegressor(features_col="features", label_col="label")
            grid = ParamGridBuilder() \
                .addGrid(xgb.max_depth, config.PARAM_GRIDS["XGBoost"]["max_depth"]) \
                .addGrid(xgb.learning_rate, config.PARAM_GRIDS["XGBoost"]["learning_rate"]) \
                .addGrid(xgb.n_estimators, config.PARAM_GRIDS["XGBoost"]["n_estimators"]) \
                .build()
            print("[MODEL] XGBoost model and hyperparameter grid constructed.")
            return xgb, grid

    def run_comparison(self, model_tuples):
        # Cross validation for each model in the comparison queue
        best_models = {}
        for name, (model, grid) in model_tuples:
            print(f"[MODEL] Training {name} with K-Fold Cross Validation...")
            cv = CrossValidator(estimator=model, estimatorParamMaps=grid, 
                                evaluator=self.evaluator, numFolds=config.NUM_FOLDS)
            best_models[name] = cv.fit(self.train_data)
        return best_models