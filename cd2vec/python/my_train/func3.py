def prediction():
	from sklearn.model_selection import KFold 
	from sklearn.model_selection import cross_val_score 
	from sklearn.model_selection import cross_val_predict 
	kfold = KFold(n_splits=10, random_state=22) 
	xyz=[]
	accuracy=[]
	std=[]
	classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']
	models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]
	for i in models:
	    model = i
	    cv_result = cross_val_score(model,X,Y, cv = kfold,scoring = 'accuracy')
	    cv_result=cv_result
	    xyz.append(cv_result.mean())
	    std.append(cv_result.std())
	    accuracy.append(cv_result)
	new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
	new_models_dataframe2
