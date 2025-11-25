import mlflow

mlflow.set_tracking_uri("http://ec2-65-2-75-98.ap-south-1.compute.amazonaws.com:5000")

with mlflow.start_run() as run:
    mlflow.log_metric("test_metric", 1.0)
    mlflow.log_param("test_param", "hello")


print("Done:", run.info.run_id)
