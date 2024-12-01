import pandas as pd
# from hsfs.feature_group import FeatureGroup
import hopsworks
from great_expectations.core import ExpectationSuite

from ..settings import SETTINGS


def push_feature_groupe_to_fs(
    data: pd.DataFrame,
    validation_expectation_suite: ExpectationSuite,
    feature_group_version: int):
    """
    Push data to feature store.
    """
    # Connect to feature store.
    project = hopsworks.login(
        api_key_value=SETTINGS["FS_API_KEY"], project=SETTINGS["FS_PROJECT_NAME"]
    )

    feature_store = project.get_feature_store()

    # Create feature group.
    energy_feature_group = feature_store.get_or_create_feature_group(
        name="btc_forcasting",
        version=feature_group_version,
        description="bitcoin/usd forcasting.",
        primary_key=["volume", "volume_ma7", "log_returns","ma7"],
        event_time="timestamp",
        online_enabled=False,
        expectation_suite=validation_expectation_suite,
    )

    # Upload data.
    energy_feature_group.insert(
        features=data,
        overwrite=False,
        write_options={
            "wait_for_job": True,
        },
    )

    return energy_feature_group