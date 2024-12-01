from feature_pipeline import utils

from feature_pipeline.etl.etl import process_crypto_data, get_kraken_data
from feature_pipeline.etl.load import push_feature_groupe_to_fs
from feature_pipeline.etl.validation import build_expectation_suite
from great_expectations.validator.validator import Validator
from great_expectations.dataset import PandasDataset


logger = utils.get_logger(__name__)


def main():
    symbol = "BTCUSD"  # Bitcoin/USD pair
    interval = 1440  # Daily data

    logger.info(f"Extracting data from API.")
    df, metadata = get_kraken_data(symbol, interval)
    
    logger.info("Successfully extracted data from API.")

    fg_version = 1
    #
    if df is None:
        raise ValueError("No data returned from Kraken API")

    # Process data
    df = process_crypto_data(df)
    logger.info("Successfully transformed data.")


    logger.info("Building validation expectation suite.")
    # Load DataFrame as a Great Expectations Dataset
    ge_df = PandasDataset(df)
    # Attach expectations to the DataFrame
    expectation_suite = build_expectation_suite()
    validation_result = ge_df.validate(expectation_suite=expectation_suite)
    logger.info("Successfully built validation expectation suite.")

    if validation_result["success"]:
        logger.info(f"loading it to the feature store.")
        fg = push_feature_groupe_to_fs(df, 
                                       expectation_suite, 
                                       fg_version)
        logger.info("Successfully validated data and loaded it to the feature store.")
    else:
        print("Validation failed!")

    metadata["feature_group_version"] = fg_version
    logger.info(f"Wrapping up the pipeline.")
    utils.save_json(metadata, file_name="feature_pipeline_metadata.json")
    logger.info("Done!")

    return metadata


if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement.
    main()


# TODO: Add hydra for global variables