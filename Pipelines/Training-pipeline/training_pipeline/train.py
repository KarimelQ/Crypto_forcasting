import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sktime.forecasting.base import ForecastingHorizon

from training_pipeline.data import load_dataset_from_feature_store
from sklearn.linear_model import LinearRegression
from sktime.forecasting.compose import make_reduction
from training_pipeline.settings import SETTINGS
import hopsworks
from pathlib import Path
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
import joblib
import os
import numpy as np
import training_pipeline.utils as utils

def attach_best_model_to_feature_store(
    model,
    metrics,
    feature_view_version: int,
    training_dataset_version: int,
    best_model_local_path: str,
) -> int:
    """Adds the best model artifact to the model registry."""

    project = hopsworks.login(
        api_key_value=SETTINGS["FS_API_KEY"], project=SETTINGS["FS_PROJECT_NAME"])

    fs = project.get_feature_store()

    feature_view = fs.get_feature_view(
        name="crypto_forcasting_view", version=feature_view_version
    )

    input_schema = Schema([{'type': 'tensor_schema', 'shape': [4, 1], 'description': 'input data to the model'}])
    output_schema = Schema([{'type': 'tensor_schema', 'shape': [1, 1]}])

    model_schema = ModelSchema(
        input_schema=input_schema, 
        output_schema=output_schema,
    )

    model_dir = "forcasting_model"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)

    # Convert the model schema to a dictionary
    model_schema.to_dict()

    joblib.dump(model, model_dir + '/forcaster_model.pkl')
    mr = project.get_model_registry()

    # Create a Python model in the model registry
    forcaster_model = mr.python.create_model(
        name="LinearRegression_forcasting_model", 
        metrics=metrics,
        model_schema=model_schema,
        input_example=np.array([[1, 2, 3, 4]]).reshape(4, 1),
        description="BTC Forcaster",
    )

    forcaster_model.save(model_dir)
    return forcaster_model.version


def train(model_path: str, feature_view_version: int, training_dataset_version: int):
    # Execute when the module is not initialized from an import statement.
    y_train, y_test, X_train, X_test = load_dataset_from_feature_store(feature_view_version, training_dataset_version)

    fh = ForecastingHorizon(range(1, len(y_test) + 1), is_relative=True)

    # Create a forecaster with make_reduction
    forecaster = make_reduction(
        estimator=LinearRegression(),  # Base estimator (regressor)
        strategy="recursive"  # Use recursive forecasting strategy
    )

    # Set up experiment in MLflow
    mlflow.set_experiment("Basic Training Experiment")
    with mlflow.start_run():

        # Fit the forecaster
        forecaster.fit(y_train, X=X_train)

        # # Predict on the test set
        y_pred = forecaster.predict(fh, X=X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print("Mean Squared Error (MSE):", mse)

        mlflow.log_metric("mse", mse, step=1)
        mlflow.log_metric("mse", mse, step=2)

        mlflow.log_metric("mae", mae, step=1)
        mlflow.log_metric("mae", mae, step=2)

        # Log the model at the end of training
        mlflow.sklearn.log_model(forecaster, "final_model")

    metrics = {
        "mse": mse,
        "mae": mae,
    }
    ## Need to return mse mae for corresponding epoch
    return forecaster, metrics


def main():
    model_path = "./output/best_model.pkl"
    # Training model
    model, metrics =  train(model_path, 1, 1)

    # attach_best_model_to_feature_store(1, 1, model_path + '/' + model_name)
    model_version = attach_best_model_to_feature_store(model, metrics, 1, 1, model_path)


    metadata = {"model_version": model_version}
    utils.save_json(metadata, file_name="train_metadata.json")

if __name__ == '__main__':
    main()


"""
TODO: 
- Get test set and perform 1 week (as example) forcasting ->
- Make plot in mlflow comparing models
"""