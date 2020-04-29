def preprocessing():
	subs  = df_weo['Subject Descriptor'].unique()[:-1]
	df_weo_agg = df_weo[['Country']][df_weo['Country'].duplicated()==False].reset_index(drop=True)
	for sub in subs[:]:
	    df_tmp = df_weo[['Country', '2019']][df_weo['Subject Descriptor']==sub].reset_index(drop=True)
	    df_tmp = df_tmp[df_tmp['Country'].duplicated()==False].reset_index(drop=True)
	    df_tmp.columns = ['Country', sub]
	    df_weo_agg = df_weo_agg.merge(df_tmp, on='Country', how='left')
	df_weo_agg.columns = [''.join (c if c.isalnum() else '_' for c in str(x)) for x in df_weo_agg.columns]
	df_weo_agg.columns
	df_weo_agg['Country_Region'] = df_weo_agg['Country']
	df_weo_agg.head()
