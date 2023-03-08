class DataFrameScaler():
    df_data_types: dict
    std_dev_df: DataFrame
    norm_dev_df: DataFrame
    scaled_df_columns: list[str]

    def __init__(
        self, 
        df: DataFrame = None,   # the Pandas DataFrame to be scaled
        filter_columns: list[str] = None   # DataFrame columns to ignore
        ) -> None:

        self.filter_columns = filter_columns
        self.df_data_types = None

        if df is not None:
            self.fit(df)