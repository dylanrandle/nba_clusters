import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import requests
import time

def get_players_df(players_url, headers):
	""" Function to fetch data from player data page and return a DataFrame

	Args:
		players_url (str): URL to NBA players metadata.
		headers (dict): headers to use in HTTP request

	Returns:
		DataFrame: Pandas object containing all relevant player metadata

	"""
	players_json=json.loads(requests.get(players_url, headers=headers).content)
	players_result_set=players_json['resultSets'][0]
	df_players=pd.DataFrame.from_records(players_result_set['rowSet'], columns=players_result_set['headers'])
	df_players.TO_YEAR=df_players.TO_YEAR.astype(int)
	return df_players

def get_player_career_reg_season_stats(player_id, player_data_url, headers):
	""" Function to fetch player's lifetime regular season stats
	
	Args:
		player_id (int): identifier for a NBA player
		player_data_url (str): base URL for NBA player-level stats
		headers (dict): headers to use in HTTP request

	Returns:
		DataFrame: Pandas object containing player's regular season stats (lifetime)

	"""
	player_data_json=json.loads(requests.get(player_data_url+str(player_id), headers=headers).content)
	career_totals=player_data_json['resultSets'][1]
	df_career_totals=pd.DataFrame.from_records(career_totals['rowSet'], columns=career_totals['headers'])
	df_career_totals.PLAYER_ID=df_career_totals.PLAYER_ID.astype(int)
	return df_career_totals

def get_player_stats(df_players, url, headers):
	"""Function to get individual player stats

	Args:
		df_players (DataFrame): DataFrame containing NBA players metadata

	Returns:
		DataFrame: Pandas object containing NBA players metadata as well as statistics
	"""
	for i, pid in enumerate(df_players['PERSON_ID']):
		if i==0:
			df_stats=get_player_career_reg_season_stats(pid, url, headers)
		else:
			df_stats=df_stats.append(
				get_player_career_reg_season_stats(pid, url, headers)
			)
		print('i={} Added player stats for ID={}'.format(i, pid))
		time.sleep(2) # sleep so we don't get blocked

	return df_players.merge(df_stats, left_on="PERSON_ID", right_on="PLAYER_ID", how='left')

def cluster(players_df, columns):
	"""Function to run KMeans on a DataFrame, using specified columns as features

	Args:
		players_df (DataFrame): DataFrame containing NBA players metadata, stats, and additional features
		columns (list): list containing all of the corresponding column headers to use as features

	Returns:
		dict: Dictionary of clusters (indexed by cluster number). Each cluster is a list of player dicts, containing basic information.
	"""
	optimal_n=None
	optimal_clusters=None
	optimal_clusterer=None
	optimal_silhouette=-99
	for n in range(2,9):
		clusterer=KMeans(n_clusters=n)
		cluster_labels=clusterer.fit_predict(players_df[columns])
		avg_silhouette=silhouette_score(players_df[columns], cluster_labels)
		print('The avg silhouette score for {} clusters is {}'.format(n, avg_silhouette))
		if avg_silhouette > optimal_silhouette:
			optimal_silhouette=avg_silhouette
			optimal_clusterer=clusterer
			optimal_clusters=cluster_labels
			optimal_n=n
	print('Returning optimal clusters found with n={}'.format(optimal_n))
	clusters = {n: [] for n in range(optimal_n)}
	for i, label in enumerate(optimal_clusters):
		clusters[label].append(
			dict(
				player_id=players_df.iloc[i]['PERSON_ID'],
				first_name=players_df.iloc[i]['DISPLAY_LAST_COMMA_FIRST'].split()[-1],
				last_name=players_df.iloc[i]['DISPLAY_LAST_COMMA_FIRST'].split()[0],
				)
			)
	return clusters

def main():
	PLAYERS_URL="https://stats.nba.com/stats/commonallplayers?LeagueId=00&Season=2016-17&IsOnlyCurrentSeason=0"
	PLAYER_DATA_URL="https://stats.nba.com/stats/playercareerstats?PerMode=PerGame&PlayerID="
	HTML_HEADERS=requests.utils.default_headers()
	HTML_HEADERS['User-Agent']="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36"
	
	df_players=get_players_df(PLAYERS_URL, HTML_HEADERS)
	# for the sake of computation, network capacity, and statistics
	# we will take a small sample from players who played last year
	n_samples=50
	df_players=df_players.query("TO_YEAR>=2018 & GAMES_PLAYED_FLAG=='Y'").sample(n=n_samples)
	print('Got {} players playing up to 2018. Change this value by altering the variable n_samples.'.format(len(df_players)))

	df_players=get_player_stats(df_players, PLAYER_DATA_URL, HTML_HEADERS)
	clusters=cluster(df_players, ['PTS', 'FGM', 'FG_PCT', 'FG3M', 'FG3_PCT', 'FT_PCT', 'OREB', 'AST', 'STL', 'BLK', 'TOV', 'DREB'])
	print('Clusters:\n',clusters)

if __name__=='__main__':
	main()