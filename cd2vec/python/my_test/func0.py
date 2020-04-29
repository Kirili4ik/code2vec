def fit_rfmodelonfulldata_on_all_data_from_the_training_data():
	
	rf_model_on_full_data = RandomForestRegressor(n_estimators = 1000, random_state = 1)rf_model_on_full_data.fit(X,y)
	
