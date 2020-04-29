def just_uncomment_them():
	test_data_path = '../input/test.csv'test_data = pd.read_csv(test_data_path)test_X = test_data[features]test_preds = rf_model_on_full_data.predict(test_X)output = pd.DataFrame({'Id': test_data.Id,
	                      'SalePrice': test_preds})
	output.to_csv('submission.csv', index=False)
