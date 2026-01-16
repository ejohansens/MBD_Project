from pyspark.sql import functions as F

class ResultsManager:
    def __init__(self, spark):
        self.spark = spark

    def save_metrics_to_csv(self, metrics_list, path):
        # Create DataFrame from results
        df = self.spark.createDataFrame(metrics_list, ["Model", "R2", "RMSE"])
        
        # Save to hdfs
        df.coalesce(1).write.csv(
            path, 
            header=True, 
            mode="overwrite"
        )
        print(f"[STORAGE] Metrics CSV folder saved to HDFS: {path}")

    def save_summary_report(self, metrics_list, total_games, path):
        report_lines = [
            "Final Model Comparison Report",
            f"Games in Study: {total_games}",
            "Model\t\tR2\t\tRMSE" 
        ]
        for name, r2, rmse in metrics_list:
            report_lines.append(f"{name}\t{r2:.4f}\t{rmse:.4f}")
        
        report_str = "\n".join(report_lines)
        report_df = self.spark.createDataFrame([(report_str,)], ["report"])
        
        # Save to hdfs
        report_df.coalesce(1).write.mode("overwrite").text(path)
        return report_str