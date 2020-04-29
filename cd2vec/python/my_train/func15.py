def modelling():
	train_df[train_df.matchType.str.contains('normal')].groupby(['matchType']).count()
