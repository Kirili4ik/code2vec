def plotting():
	
	df[['Aerial Battles Won','Duels Won','Recoveries','Tackle Success %']]= df[['Aerial Battles Won','Duels Won','Recoveries','Tackle Success %']].fillna(0).astype(int)cm = sns.light_palette('orange', as_cmap=True)
	df.groupby('Club')['Aerial Battles Won','Duels Won','Tackle Success %','Recoveries'].sum().sort_values(by='Recoveries',ascending=False).head(20).style.background_gradient(cmap=cm)
