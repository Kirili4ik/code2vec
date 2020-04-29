def create_x():
	
	import pandas as pd
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.metrics import mean_absolute_error
	from sklearn.model_selection import train_test_split
	from sklearn.tree import DecisionTreeRegressor
	from learntools.core import *iowa_file_path = '../input/train.csv'home_data = pd.read_csv(iowa_file_path)y = home_data.SalePricefeatures = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
	X = home_data[features]train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)iowa_model = DecisionTreeRegressor(random_state=1)iowa_model.fit(train_X, train_y)val_predictions = iowa_model.predict(val_X)
	val_mae = mean_absolute_error(val_predictions, val_y)
	print('Validation MAE when not specifying max_leaf_nodes: {:,.0f}'.format(val_mae))iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
	iowa_model.fit(train_X, train_y)
	val_predictions = iowa_model.predict(val_X)
	val_mae = mean_absolute_error(val_predictions, val_y)
	print('Validation MAE for best value of max_leaf_nodes: {:,.0f}'.format(val_mae))rf_model = RandomForestRegressor(random_state=1)
	rf_model.fit(train_X, train_y)
	rf_val_predictions = rf_model.predict(val_X)
	rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)print('Validation MAE for Random Forest Model: {:,.0f}'.format(rf_val_mae))
	
