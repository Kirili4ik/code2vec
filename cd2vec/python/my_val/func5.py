def check_your_answer():
	reviewer_mean_ratings = reviews.groupby('taster_name')['points'].mean()
	print(reviewer_mean_ratings)q5.check()
