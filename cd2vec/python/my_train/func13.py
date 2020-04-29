def linear_algebra():
	import numpy as np  
	import pandas as pd  
	from datetime import datetimefrom scipy.stats import skew   for some statistics
	from scipy.special import boxcox1p
	from scipy.stats import boxcox_normmaxfrom sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
	from sklearn.ensemble import GradientBoostingRegressor
	from sklearn.svm import SVR
	from sklearn.pipeline import make_pipeline
	from sklearn.preprocessing import RobustScaler
	from sklearn.model_selection import KFold, cross_val_score
	from sklearn.metrics import mean_squared_errorfrom mlxtend.regressor import StackingCVRegressorfrom xgboost import XGBRegressor
	from lightgbm import LGBMRegressor
