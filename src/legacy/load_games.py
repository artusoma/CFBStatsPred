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
    offeff: float
    defeff: float
    win_p: float = .5
    record_history: Dict[int, List[int, int]] = {}
    offeff_history: Dict[int, Dict[int, float]] = {} #Year => week => offeff
    defeff_history: Dict[int, Dict[int, float]] = {} #Year => week => defeff

    def get_record(self, year: int):
        wins = self.record_history[year][0]
        losses = self.record_history[year][1]
        return wins/(wins + losses)

@dataclass
class Game:
    week: int
    home: str
    away: str
    home_score: int
    away_score: int

##Large season class used to storing team statistics
class Season:
    """
    Some things to do:
        1. Probably move pulling games to some sort of iterable function instead of reusing same code
        2. Move season to its own class. Then, have different models inherit this class. 
        3. Keep history, winning percentage, etc. in Team. Maybe move this to a class. Eventually, maybe Game should
        also be a class once we being storing drive data. 

        Some models
            1. Drive by drive
            2. Bayesian
            3. Machine learning
    """

    def __init__(self, offinit: float = 28., definit: float = 28.):
        ##Initalize some variables
        self.schedules = {} ##Schedules for each year. Maps year => list of games
        self.teams = {}     ##Maps team name => Team obj,
        self.team_history = {} ##Maps team name => [(offeff, defeff, pythag), ...] for each game
        self.team_winning = {} ##Maps team name => (wins, losses)
        self.season_ends = {}  ##Maps team name => [(offeff, defeff), ...] but only for end of season

        self.offinit = offinit
        self.definit = definit
        return 

    def load_season_schedule(self, year: int, fpath: str) -> None:
        """Loads a season from a CSV into a season object.
        """
        games = pd.read_csv(fpath)
        games = games[                      #limit to p5 for now
            games.home_conference.isin(['Pac-12', 'Big Ten', 'Big 12', 'SEC', 'ACC']) &
            games.away_conference.isin(['Pac-12', 'Big Ten', 'Big 12', 'SEC', 'ACC']) 
        ].reset_index()

        ##Iterate through games and create game objects for each one
        season_games = {}
        for row_idx in range(games.shape[0]):
            week = games.loc[row_idx, 'week']
            home_team = games.loc[row_idx, 'home_team']
            away_team = games.loc[row_idx, 'away_team']
            home_score = games.loc[row_idx, 'home_points']
            away_score = games.loc[row_idx, 'away_points']
            gtemp = Game(week, home_team, away_team, home_score, away_score)
        
            if week in season_games: ##See if we already have that week on file; if not create it
                season_games[week].append(gtemp)
            else:
                season_games[week] = [gtemp]

        self.schedules[year] = season_games
        return

    def init_teams(self, fpaths: str) -> None:
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
        self.teams = {team:Team(team, self.offinit, self.definit) for team in unique_teams}
        self.team_history = {team:[] for team in unique_teams}
        self.season_ends = {team:[] for team in unique_teams}
        return 

    def train_elo(self, year: int, K: float = .2):
        ##Iterate through each week in schedule
        for week in self.schedules[year]:
            games_in_week = self.schedules[year][week] ##Grab that weeks games

            ##Iterate through weekly games to update
            for game in games_in_week:
                home = self.teams[game.home] ##Grab team objects from team names
                away = self.teams[game.away]
                ##Adjust home offeff
                expected_home_offense = (home.offeff + away.defeff)/2
                home.offeff = home.offeff + K*(game.home_score-expected_home_offense)
                ##Adjust away offeff
                expected_away_offense = (away.offeff + home.defeff)/2
                away.offeff = away.offeff + K*(game.away_score-expected_away_offense)
                home.defeff = home.defeff + K*(game.away_score-expected_away_offense) ##Adjust home defeff
                away.defeff = away.defeff + K*(game.home_score-expected_home_offense) ##Adjust away defeff
                ##Now, update winning percentages
                home.win_p = pythag_win_percentage(home)
                away.win_p = pythag_win_percentage(away)

                ##Add to histories
                self.team_history[game.home].append((home.win_p, home.offeff, home.defeff))
                self.team_history[game.away].append((away.win_p, away.offeff, away.defeff))
        return

    def evaluate_predicting_power(self, year: int, power: float = 2) -> float:
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

    def init_next_year_scores(self, weights: np.ndarray) -> None:
        for team in self.teams:
            team_obj = self.teams[team]
            prev_offeffs = np.array([t[0] for t in self.season_ends[team]])
            prev_defeffs = np.array([t[1] for t in self.season_ends[team]])

            if len(prev_offeffs) < 5:
                prev_offeffs = np.pad(prev_offeffs, (5-len(prev_offeffs), 0), mode='constant', constant_values=self.offinit)
                prev_defeffs = np.pad(prev_defeffs, (5-len(prev_defeffs), 0), mode='constant', constant_values=self.definit)
            if len(prev_defeffs) > 5:
                prev_offeffs = prev_offeffs[-5:]
                prev_defeffs = prev_defeffs[-5:]

            new_offeff = np.dot(prev_offeffs, weights)/np.sum(weights)
            new_defeff = np.dot(prev_defeffs, weights)/np.sum(weights)

            team_obj.offeff = new_offeff
            team_obj.defeff = new_defeff
        return 

    def save_ending_effs(self) -> None:
        for team in self.teams:
            team_obj = self.teams[team]
            self.season_ends[team].append( (team_obj.offeff, team_obj.defeff) )
        return 

#######################################
##Define some statistical calculations
def pythag_win_percentage(team: Team, power: float = 2):
    """Pythagorean calcuation of average win percentage of a team
    """
    offeff = team.offeff
    defeff = team.defeff
    return offeff**power/(offeff**power+defeff**power)

def log5_win_prob(team1: Team, team2: Team, home_adv: float = 0.1):
    """Log5 calculation of probability than team1 beats team2
    """
    p1 = team1.win_p
    p2 = team2.win_p
    return (p1-p1*p2)/(p1+p2-2*p1*p2) + home_adv

def score_dist():
    """Model win probabilties -> score differential"""
    return 



##An example of how to use the Season object
if __name__ == "__main__":
    ##Initalize season and load team + season data
    season = Season()
    season.init_teams(["../data/2019season.csv"])
        
    for year in [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]:
        season.load_season_schedule(year, f"../data/{year}season.csv")