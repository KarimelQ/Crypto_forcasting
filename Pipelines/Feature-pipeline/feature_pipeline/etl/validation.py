from great_expectations.core import ExpectationSuite, ExpectationConfiguration


def build_expectation_suite() -> ExpectationSuite:
    """
    Builder used to retrieve an instance of the validation expectation suite.
    """

    crypto_forcasting = ExpectationSuite(
        expectation_suite_name="btc_forcasting_suite"
    )

    # Columns.
    crypto_forcasting.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_columns_to_match_ordered_list",
            kwargs={
                "column_list": [
                    "timestamp",
                    "volume",
                    "volume_ma7",
                    "log_returns",
                    "ma7",
                    "close",
                ]
            },
        )
    )
    crypto_forcasting.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_column_count_to_equal", kwargs={"value": 6}
        )
    )
    
    # Check for no NaN values.
    for column in ["timestamp", "volume", "volume_ma7", "log_returns", "ma7", "close"]:
        crypto_forcasting.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": column},
            )
        )

    return crypto_forcasting


# TODO: 
# Add statistics of features 