import pandas as pd
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

##Define a season and game class
@dataclass
class Team:
    name: str
    offeff: float = 28
    defeff: float = 28
    win_p: float = .5

@dataclass
class Game:
    week: int
    home: Team
    away: Team
    home_score: int
    away_score: int

##Large season class used to storing team statistics
class Season:
    def __init__(self):
        ##Initalize some variables
        self.schedules = {} ##Schedules for each year. Maps year => list of games
        self.teams = {} 

        self.team_history = {} ##Maps team => [(offeff, defeff, pythag)]
        self.team_winning = {} ##Maps team => (wins, losses)
        return 

    def load_season_schedule(self, year: int, fpath: str) -> None:
        """Loads a season from a CSV into a season object.
        """
        games = pd.read_csv(fpath)
        games = games[                      #limit to p5 for now
            games.home_conference.isin(['Pac-12', 'Big Ten', 'Big 12', 'SEC', 'ACC']) &
            games.away_conference.isin(['Pac-12', 'Big Ten', 'Big 12', 'SEC', 'ACC']) 
        ].reset_index()

        season_games = {}
        ##Iterate through games and create game objects for each one
        for row_idx in range(games.shape[0]):
            week = games.loc[row_idx, 'week']
            home_team = games.loc[row_idx, 'home_team']
            away_team = games.loc[row_idx, 'away_team']
            home_score = games.loc[row_idx, 'home_points']
            away_score = games.loc[row_idx, 'away_points']
        
            gtemp = Game(week, home_team, away_team, home_score, away_score)
            if week in season_games:
                season_games[week].append(gtemp)
            else:
                season_games[week] = [gtemp]

        self.schedules[year] = season_games
        return

    def init_teams(self, fpaths):
        unique_teams = []
        for f in fpaths:
            games = pd.read_csv(f)
            games = games[                      #limit to p5 for now
                games.home_conference.isin(['Pac-12', 'Big Ten', 'Big 12', 'SEC', 'ACC']) &
                games.away_conference.isin(['Pac-12', 'Big Ten', 'Big 12', 'SEC', 'ACC']) 
            ].reset_index()
            unique_teams.extend(games.home_team)
            unique_teams.extend(games.away_team)
        unique_teams = sorted(list(set(unique_teams)))
        self.teams = {team:Team(team) for team in unique_teams}
        self.team_history = {team:[] for team in unique_teams}
        return 

    def train_elo(self, year: int, K: float = .2):
        for week in self.schedules[year]:
            games_in_week = self.schedules[year][week]

            for game in games_in_week:
                home = self.teams[game.home]
                away = self.teams[game.away]

                ##Adjust home offeff
                expected_home_offense = (home.offeff + away.defeff)/2
                home.offeff = home.offeff + K*(game.home_score-expected_home_offense)

                ##Adjust away offeff
                expected_away_offense = (away.offeff + home.defeff)/2
                away.offeff = away.offeff + K*(game.away_score-expected_away_offense)

                ##Adjust home defeff
                home.defeff = home.defeff + K*(game.away_score-expected_away_offense)

                ##Adjust away defeff
                away.defeff = away.defeff + K*(game.home_score-expected_home_offense)

                ##Now, update winning percentages
                home.win_p = pythag_win_percentage(home)
                away.win_p = pythag_win_percentage(away)

                self.team_history[game.home].append(home.defeff)
                self.team_history[game.away].append(away.defeff)

        return season

    def evaluate_predicting_power(self, year: int, power: float = 2):
        ncorrect = 0
        total = 0
        for team_name in self.teams:
            team_obj = self.teams[team_name]
            team_obj.win_p = pythag_win_percentage(team_obj, power=power)

        for week in self.schedules[year]:
            games_in_week = self.schedules[year][week]
            for game in games_in_week:
                team1 = self.teams[game.home]
                team2 = self.teams[game.away]
                #print(f"{game.home} vs {game.away} :: {game.home_score}-{game.away_score} :: Predicted {log5_win_prob(team1, team2)}")

                if (game.home_score > game.away_score) & (log5_win_prob(team1, team2) > .5):
                    ncorrect += 1
                if (game.home_score < game.away_score) & (log5_win_prob(team1, team2) < .5):
                    ncorrect += 1
                total += 1

        return ncorrect/total
        
    def evaluate_pythag_power(self, year: int, power: float = 2):
        self.team_winning = {team:[0,0] for team in self.teams}

        ##Compute updated winning percentage with given power
        for team_name in self.teams:
            team_obj = self.teams[team_name]
            team_obj.win_p = pythag_win_percentage(team_obj, power=power)

        ##Iterate through games and see which teams won and lost
        for week in self.schedules[year]:
            games_in_week = self.schedules[year][week]
            for game in games_in_week:
                if (game.home_score > game.away_score):
                    self.team_winning[game.home][0] += 1
                    self.team_winning[game.away][1] += 1
                else:
                    self.team_winning[game.home][1] += 1
                    self.team_winning[game.away][0] += 1

        ##Compare these values to that given by pythag
        pythag_values = []
        observed_values = []
        for team_name in self.teams:
            pythag_values.append(self.teams[team_name].win_p)
            observed_values.append(self.team_winning[team_name][0]/(self.team_winning[team_name][0]+self.team_winning[team_name][1]))
        return np.mean( (np.array(pythag_values) - np.array(observed_values))**2 )


#######################################
##Define some statistical calculations
def pythag_win_percentage(team: Team, power: float = 2):
    """Pythagorean calcuation of average win percentage of a team
    """
    offeff = team.offeff
    defeff = team.defeff
    return offeff**power/(offeff**power+defeff**power)

def log5_win_prob(team1: Team, team2: Team):
    """Log5 calculation of probability than team1 beats team2
    """
    p1 = team1.win_p
    p2 = team2.win_p
    return (p1-p1*p2)/(p1+p2-2*p1*p2) + .10

def score_dist():
    """Model win probabilties -> score differential"""
    return 

#########################################
##Main script
if __name__ == "__main__":
    for k in [0.005, 0.01, 0.05]:
        ##Initalize season and load team + season data
        season = Season()
        season.init_teams(["../data/2019season.csv"])
        
        for year in [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]:
            season.load_season_schedule(year, f"../data/{year}season.csv")

        ##Train for every year you want
        for year in [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]:
            season.train_elo(year=year, K=k)

        for p in [7]:
            mse = season.evaluate_pythag_power(year=2018, power=p)
            print(k, p, mse)

        pcorrect = season.evaluate_predicting_power(year=2019, power=7)
        print(k, pcorrect)

    plt.plot(range(len(season.team_history['Michigan State'])), season.team_history['Michigan State'], c='green')
    plt.plot(range(len(season.team_history['Michigan'])), season.team_history['Michigan'])
    plt.plot(range(len(season.team_history['Alabama'])), season.team_history['Alabama'])
    plt.plot(range(len(season.team_history['Indiana'])), season.team_history['Indiana'])
    plt.plot(range(len(season.team_history['LSU'])), season.team_history['LSU'])
    plt.show()