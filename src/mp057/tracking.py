import hashlib
import mlflow
import mlflow.sklearn


def calculate_data_hash(data_path):
    """Calculate SHA-256 hash of the original training dataset."""
    sha256 = hashlib.sha256()

    with open(data_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def setup_mlflow(tracking_uri, experiment_name):
    """Configure MLflow experiment tracking."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_experiment(model, model_name, params, metrics, data_hash):
    """Log model parameters, metrics, hash and model artifact."""

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("data_sha256", data_hash)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )

        return mlflow.active_run().info.run_id
