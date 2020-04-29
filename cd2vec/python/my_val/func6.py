def define_and_fit_model():
	
	model = RandomForestRegressor(n_estimators=100, random_state=0)
	model.fit(final_X_train, y_train)preds_valid = model.predict(final_X_valid)
	print('MAE (Your appraoch):')
	print(mean_absolute_error(y_valid, preds_valid))
