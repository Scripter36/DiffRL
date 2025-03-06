import os
import mlflow
import queue
import threading
import json
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(override=True)  # This loads variables from .env into os.environ

import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)

# Set the MLFlow tracking URI if the environment variable is provided.
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLFlow tracking URI: {tracking_uri}")

mlflow.config.enable_async_logging()

# thread queue
_save_queue = queue.Queue()
_save_worker_thread = None

def _mlflow_save_worker():
    while True:
        task = _save_queue.get()
        if task is None:
            break
        checkpoint_model, filename = task
        active_run = get_current_run()
        # save the model
        mlflow.pytorch.log_model(checkpoint_model, filename, run_id=active_run.info.run_id)
        # https://github.com/mlflow/mlflow/issues/10802
        # remove the existing model history, to prevent overflow (8000 char limit)
        run = mlflow.get_run(active_run.info.run_id)
        if 'mlflow.log-model.history' in run.data.tags:
            history = json.loads(run.data.tags['mlflow.log-model.history'])
            # delete the item with same artifact_path, while remaining the latest one
            same_model_history = [item for item in history if item['artifact_path'] == filename]
            other_history = [item for item in history if item['artifact_path'] != filename]
            if len(same_model_history) > 1:
                same_model_history.sort(key=lambda x: datetime.strptime(x['utc_time_created'], '%Y-%m-%d %H:%M:%S.%f'), reverse=True)
                same_model_history = same_model_history[:1]
            history = same_model_history + other_history
            get_mlflow_client().set_tag(run.info.run_id, 'mlflow.log-model.history', json.dumps(history), True)
        _save_queue.task_done()

def start_save_worker():
    global _save_worker_thread
    if _save_worker_thread is None or not _save_worker_thread.is_alive():
        _save_worker_thread = threading.Thread(target=_mlflow_save_worker, daemon=True)
        _save_worker_thread.start()

def enqueue_model_save(checkpoint_model, filename):
    # Ensure the save worker thread is running.
    start_save_worker()
    _save_queue.put((checkpoint_model, filename))

def stop_save_worker():
    # Optionally, stop the worker thread gracefully.
    _save_queue.put(None)
    if _save_worker_thread is not None:
        _save_worker_thread.join()

_mlflow_client = None
def get_mlflow_client():
    global _mlflow_client
    if _mlflow_client is None:
        _mlflow_client = mlflow.MlflowClient()
    return _mlflow_client

_current_run = None
def get_current_run() -> 'mlflow.ActiveRun':
    global _current_run
    return _current_run

def set_current_run(run):
    global _current_run
    _current_run = run

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
