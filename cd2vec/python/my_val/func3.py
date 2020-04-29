def read_the_data():
	import pandas as pd
	from sklearn.model_selection import train_test_splitX_full = pd.read_csv('../input/train.csv', index_col='Id')
	X_test_full = pd.read_csv('../input/test.csv', index_col='Id')X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
	y = X_full.SalePrice
	X_full.drop(['SalePrice'], axis=1, inplace=True)X = X_full.select_dtypes(exclude=['object'])
	X_test = X_test_full.select_dtypes(exclude=['object'])X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
	                                                      random_state=0)
