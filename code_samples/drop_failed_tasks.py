# df is a preprocessed dataframe provided by Alibaba
df.query("status=='Terminated'", inplace=True)
df.drop(columns=['status'], inplace=True)