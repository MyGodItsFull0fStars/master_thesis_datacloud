def fit(self, df: DataFrame) -> None:
    self.df_columns = df.columns
    # preserve datatypes of dataframe
    self.df_data_types = df.dtypes.to_dict()
    self.scaled_df_columns = self.get_scaled_columns(
        df, self.filter_columns)
    self.std_dev_df = self._get_std_mean_df(df)
    self.norm_dev_df = self._get_norm_min_max_df(df)