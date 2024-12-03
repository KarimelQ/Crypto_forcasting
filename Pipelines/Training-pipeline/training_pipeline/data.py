from typing import Tuple
import hopsworks
import pandas as pd
# import wandb

from sktime.forecasting.model_selection import temporal_train_test_split

# from training_pipeline.utils import init_wandb_run
from training_pipeline.settings import SETTINGS

def prepare_data(
    data: pd.DataFrame, target: str = "close", fh: int = 24
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Structure the data for training:
    - Set the index as is required by sktime.
    - Prepare exogenous variables.
    - Prepare the time series to be forecasted.
    - Split the data into train and test sets.
    """

    # Convert timestamp column to datetime format and set it as the index
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)

    # Target variable (time series to forecast)
    y = data['close'].values.reshape(-1, 1)

    # Exogenous variables (if applicable)
    X = data[['volume', 'volume_ma7', 'log_returns', 'ma7']].values

    print("Features ", X.shape)
    print("Target ", y.shape)
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=fh)

    return y_train, y_test, X_train, X_test


def load_dataset_from_feature_store(
    feature_view_version: int, training_dataset_version: int, fh: int = 24
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load features from feature store.

    Args:
        feature_view_version (int): feature store feature view version to load data from
        training_dataset_version (int): feature store training dataset version to load data from
        fh (int, optional): Forecast horizon. Defaults to 24.

    Returns:
        Train and test splits loaded from the feature store as pandas dataframes.
    """

    project = hopsworks.login(
        api_key_value=SETTINGS["FS_API_KEY"], project=SETTINGS["FS_PROJECT_NAME"]
    )
    fs = project.get_feature_store()

    feature_view = fs.get_feature_view(
        name="crypto_forcasting_view", version=feature_view_version
    )

    data, _ = feature_view.get_training_data(
        training_dataset_version=training_dataset_version
    )

    print("data ", data)
    fv_metadata = feature_view.to_dict()
    fv_metadata["query"] = fv_metadata["query"].to_string()
    fv_metadata["features"] = [f.name for f in fv_metadata["features"]]
    fv_metadata["link"] = feature_view._feature_view_engine._get_feature_view_url(
        feature_view
    )
    fv_metadata["feature_view_version"] = feature_view_version
    fv_metadata["training_dataset_version"] = training_dataset_version


    y_train, y_test, X_train, X_test = prepare_data(data, fh=fh)

    return y_train, y_test, X_train, X_test


if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement.
    load_dataset_from_feature_store(1, 1)
