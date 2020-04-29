def plotting():
	
	ts_fare_diff = log_ts_fare - log_ts_fare.shift()
	ts_fare_diff.dropna(inplace = True)t1 = plot_line(ts_fare_diff.index,ts_fare_diff['fare_amount'],
	              'blue','Differenced log series')
	lay = plot_layout('Differenced log series')
	fig = go.Figure(data = [t1],layout=lay)
	py.iplot(fig)stationary_test(ts_fare_diff)
	
