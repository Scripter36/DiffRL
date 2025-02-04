import os
import mlflow

from dotenv import load_dotenv
load_dotenv()  # This loads variables from .env into os.environ

# Set the MLFlow tracking URI if the environment variable is provided.
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLFlow tracking URI: {tracking_uri}")

mlflow.config.enable_async_logging()

# Singleton class to manage MLflow client and active run
class MLFlowManager:
    _instance = None
    _mlflow_client = None 
    _active_run = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLFlowManager, cls).__new__(cls)
        return cls._instance

    @property
    def mlflow_client(self) -> mlflow.MlflowClient:
        if self._mlflow_client is None:
            self._mlflow_client = mlflow.MlflowClient()
        return self._mlflow_client

    @property
    def active_run(self) -> mlflow.ActiveRun:
        return self._active_run

    @active_run.setter 
    def active_run(self, run: mlflow.ActiveRun):
        self._active_run = run

mlflow_manager = MLFlowManager()

def flatten_dict(d, parent_key='', sep='.'):
    """
    Recursively flattens a nested dictionary.
    For example, {"a": {"b": 1, "c": 2}} becomes {"a.b": 1, "a.c": 2}.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def unflatten_dict(d, sep='.'):
    """
    Unflattens a dictionary.
    For example, {"a.b": 1, "a.c": 2} becomes {"a": {"b": 1, "c": 2}}.
    """
    result = {}
    for k, v in d.items():
        parts = k.split(sep)
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = v
    return result

def set_experiment_name_from_env(default_experiment_name):
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", default_experiment_name)
    mlflow.set_experiment(experiment_name)
    print(f"MLFlow experiment: {experiment_name}")
