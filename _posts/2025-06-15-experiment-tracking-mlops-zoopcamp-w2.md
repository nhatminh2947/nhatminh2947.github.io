---
title:  "Experiment Tracking with MLflow"
date:   2025-06-15 21:16+07:00
categories: [MLOps Zoomcamp 2025]
---

## Table of Contents

- [Introduction](#introduction)
- [What is MLflow?](#what-is-mlflow)
- [Why Bother Tracking Experiments, Anyway?](#why-bother-tracking-experiments-anyway)
- [Setting Up the Environment](#setting-up-the-environment)
- [Preparing the Dataset](#preparing-the-dataset)
- [Training and Logging Experiments with MLflow](#training-and-logging-experiments-with-mlflow)
- [Hyperparameter Tuning with Hyperopt and MLflow](#hyperparameter-tuning-with-hyperopt-and-mlflow)
- [Registering and Promoting Your Best Model](#registering-and-promoting-your-best-model)
- [Using Your Registered Model and Managing Model Stages](#using-your-registered-model-and-managing-model-stages)
- [Resources](#resources)

---

## Introduction

Picture this: you’re deep in a machine learning project, juggling experiments like a circus act. The last thing you want is to play detective, trying to remember if your best run was “final_final_v2” or “test123”, or where you stashed the results—on your desktop, in “new_folder_2”, or somewhere even more mysterious. And don’t even get me started on tracking down the metrics.

No one wants to live in that chaos. So, ditch the sticky notes and memory games—let MLflow track your experiments for you. Your future self will thank you!

Ready to leave the mess behind? Let’s dive into a step-by-step guide on setting up MLflow and tracking your experiments.

---

## What is MLflow?

> “MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It tackles four primary functions: tracking experiments to record and compare parameters and results, packaging code into reproducible runs, sharing and deploying models, and managing and versioning datasets.”  
> — [mlflow.org](https://mlflow.org/)

**Key MLflow Concepts:**
- **Experiment:** A collection of runs, usually grouped by a project or a specific task.
- **Run:** A single execution of your training script. Each run logs parameters, metrics, and artefacts.
- **Artefacts:** Files or objects produced during a run—like models, plots, or preprocessed data.

---

## Why Bother Tracking Experiments, Anyway?

As a data scientist or machine learning engineer, keeping your work organised is crucial—especially when collaborating with others. While tracking experiments in Excel might suffice when you’re working solo, things quickly become unmanageable in a team environment. MLflow provides a centralised, structured way to log, share, and discuss experiments, making collaboration seamless.

Reproducibility is another key challenge in machine learning. With MLflow, you can experiment freely, knowing you’ll always be able to retrace your steps—even weeks later. Every parameter, metric, and artefact is logged, so you’ll never lose track of what you did in “Experiment 1”.

Finally, MLflow enables automation of post-experiment steps, such as verifying model performance before deploying to production. This not only saves time but also reduces the risk of human error, ensuring your models are production-ready and reliable.

---

## Setting Up the Environment

First, install MLflow:

```bash
pip install mlflow
```

To run the MLflow UI locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

This will create a `mlflow.db` file in your current folder. This tells MLflow to store all artefacts, metrics, and metadata in SQLite.  
You can now access the web UI at [http://127.0.0.1:5000](http://127.0.0.1:5000). It should look something like this:

<img width="1512" alt="Screenshot 2568-06-16 at 00 48 07" src="https://github.com/user-attachments/assets/55702d26-5b77-441a-a35b-787625e90060" />

---

## Preparing the Dataset

For this tutorial, we’ll use the Green Taxi Trip Records dataset to predict trip duration.

1. **Download the data** for January, February, and March 2023 in parquet format from [here](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
2. **Preprocess the data** using the provided script:

```bash
python preprocess_data.py --raw_data_path <TAXI_DATA_FOLDER> --dest_path ./output
```

This script will:
- Load the data from `<TAXI_DATA_FOLDER>`
- Fit a `DictVectorizer` on the training set (January 2023)
- Save the preprocessed datasets and vectorizer to the `output` folder

In the `output` folder, you’ll find three datasets: `train.pkl`, `test.pkl`, and `val.pkl`.

---

## Training and Logging Experiments with MLflow

Let’s train a simple RandomForestRegressor and track everything with MLflow.

First, tell MLflow to store all experiment tracking data in the SQLite database and set an experiment called `"nyc-taxi-exp"`:

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-exp")
```

Next, define a new run using `with mlflow.start_run():`. Everything inside this block will be associated with the run.

You can start logging information about the run using:

```python
mlflow.log_param()
mlflow.log_metric()
```

Here’s a simple script to train a RandomForestRegressor and track the `data_path` parameter and `rmse` metric:

```python
import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-exp")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    with mlflow.start_run():
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_param('data_path', data_path)
        mlflow.log_metric('rmse', rmse)

if __name__ == '__main__':
    run_train()
```

Back in the UI, you’ll see a run has been generated (with a fun name like `fun-turtle-105`). Click into the run to find the `data_path` parameter and its value (`output`), as well as the `rmse` metric.

<img width="1512" alt="Screenshot 2568-06-15 at 23 24 02" src="https://github.com/user-attachments/assets/31240f1e-1c0e-4b83-868d-04c104dc5fb3" />

<img width="754" alt="Screenshot 2568-06-15 at 23 24 28" src="https://github.com/user-attachments/assets/e7114530-3e59-407a-992e-9d52f1ddbf2c" />

<img width="1496" alt="Screenshot 2568-06-15 at 23 24 41" src="https://github.com/user-attachments/assets/3b8d15f0-b98b-4b63-b82a-86145e108ca3" />

---

## Hyperparameter Tuning with Hyperopt and MLflow

Now that you can track a single experiment, imagine needing to run multiple models or perform hyperparameter tuning. How do you keep track of all those runs?

First, install `hyperopt` for hyperparameter tuning:

```bash
pip install hyperopt
```

Set the experiment as `random-forest-hyperopt`:

```python
mlflow.set_experiment("random-forest-hyperopt")
```

Define the objective function:

```python
def objective(params):
    with mlflow.start_run():
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_params(params)
        mlflow.log_metric(key='rmse', value=rmse)
        return {'loss': rmse, 'status': STATUS_OK}
```

Define the search space:

```python
search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
    'random_state': 42
}
```

Run the hyperparameter optimisation:

```python
fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=num_trials,
    trials=Trials(),
    rstate=rstate
)
```

**Full hyperparameter tuning script:**

```python
import os
import pickle
import click
import mlflow
import mlflow.sklearn
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("random-forest-hyperopt")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(params):
        with mlflow.start_run():
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_params(params)
            mlflow.log_metric(key='rmse', value=rmse)
            return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    rstate = np.random.default_rng(42)
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )

if __name__ == '__main__':
    run_optimization()
```

**Run the hyperparameter tuning:**

```bash
python hpo.py --data_path ./output --num_trials 15
```

This script automates hyperparameter tuning for a Random Forest model using Hyperopt.  
Each trial is logged in MLflow, so you can compare all runs in the MLflow UI.

After running the script, you’ll find 15 runs in the MLflow UI, each with its `rmse` metric and the parameters it was trained with.

<img width="1512" alt="Screenshot 2568-06-16 at 01 08 59" src="https://github.com/user-attachments/assets/f39eea8c-b231-42e9-a9ce-dfcf30c26387" />

---

## Registering and Promoting Your Best Model

After tuning, you’ll want to register your best model in the MLflow Model Registry for easy access and deployment.

First, set a new experiment name for this step:

```python
EXPERIMENT_NAME = 'random-forest-best-models'
mlflow.set_experiment(EXPERIMENT_NAME)
```

Let’s say you want to select the top 5 models with the best performance on the validation data. You can use `search_runs` to do that.  
More details can be found [here](https://www.mlflow.org/docs/latest/ml/search/search-runs).

```python
client = MlflowClient()
experiment = client.get_experiment_by_name('random-forest-hyperopt')
runs = client.search_runs(
    experiment_ids=experiment.experiment_id,
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=top_n,
    order_by=["metrics.rmse ASC"]
)
```

After evaluating the top 5 models on the test set, you can register the best one:

```python
run_id = '5ebc3688d10b464d841e480c44db8fac'
best_model_uri = f'runs:/{run_id}/model'
mlflow.register_model(model_uri=best_model_uri, name='nyc-taxi-regressor')
```

**Registration Script (`register_model.py`):**

```python
import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))
    
    with mlflow.start_run():
        new_params = {param: int(params[param]) for param in RF_PARAMS}
        rf = RandomForestRegressor(**new_params)
        rf.fit(X_train, y_train)
        val_rmse = root_mean_squared_error(y_val, rf.predict(X_val))
        test_rmse = root_mean_squared_error(y_test, rf.predict(X_test))
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_artifact(local_path="output/dv.pkl", artifact_path='model')
        mlflow.sklearn.log_model(sk_model=rf, artifact_path='model')

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()
    
    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        order_by=["metrics.test_rmse ASC"]
    )
    best_run = runs[0]
    best_model_uri = f'runs:/{best_run.info.run_id}/model'
    mlflow.register_model(model_uri=best_model_uri, name='nyc-taxi-regressor')

if __name__ == '__main__':
    run_register_model()
```

**Run the script:**

```bash
python register_model.py --data_path ./output --top_n 5
```

This will select the top 5 models, retrain and evaluate them, and register the best one in the MLflow Model Registry.

---

## Using Your Registered Model and Managing Model Stages

Now that your model is registered, you can load it with `MlflowClient` and manage its lifecycle by transitioning it between stages (e.g., Staging, Production).

### Transitioning Model Stages

You can move your model between stages using the MLflow UI or programmatically:

```python
from mlflow.tracking import MlflowClient
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
model_name = "nyc-taxi-regressor"
model_version = 1  # Replace with your model version
model_version_alias = "Staging"
# Transition to Staging
client.set_registered_model_alias(
    name=model_name,
    version=model_version,
    alias="Staging"
)
```

### Load model via Model Version Alias

```python
model_uri = f"models:/{model_name}@{model_version_alias}"
model = mlflow.sklearn.load_model(model_uri)
```

---

## Conclusion

Experiment tracking doesn’t have to be a headache. 
With MLflow, you can keep your machine learning projects organised, reproducible, and ready for collaboration—no more lost results or mystery folders. 
Whether you’re tuning hyperparameters, registering your best models, or managing production deployments, MLflow makes the whole process smoother and more transparent. 
Give it a go in your next project, and you’ll wonder how you ever managed without it!

---

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLOps Zoomcamp 2025: Experiment Tracking](https://github.com/nhatminh2947/mlops-zoomcamp/tree/master/02-experiment-tracking)
- [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main)
