# instances is a pd.Series that contains the instance number of a task
instance_indices = instances.apply(lambda x: get_index_pos(x <= ranges))
dummies = pd.get_dummies(instance_indices)
dummies.rename(
    columns={
        col: f'inst_clust_{col}' for col in dummies.columns
        }, 
    inplace=True)
# training_df is a pd.DataFrame that contains the training dataset
inst_df = training_df.join(dummies)