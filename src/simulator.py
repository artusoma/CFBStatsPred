import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import copy

##Define a season and game class
@dataclass
class Team:
    name: str
    offeff: float
    defeff: float
    win_p: float = .5
    record_history: Dict[int, List[int]] = field(default_factory=dict) #Year => [wins, losses]
    stat_history: Dict[int, Dict[int, Tuple[float, float, float]]] = field(default_factory=dict)  #Year => week => winp, off, def
    conf: str = None # "P5", "G5", "Other" going to be used to set effs & K, maybe others

    def get_record(self, year: int) -> float:
        wins = self.record_history[year][0]
        losses = self.record_history[year][1]
        return wins/(wins + losses)

    def add_history(self, year: int, week: int)-> None:
        if year not in self.stat_history:
            self.stat_history[year] = {}
        self.stat_history[year][week] = (self.win_p, self.offeff, self.defeff)
        return 

    def update_wl(self, year: int, won: bool) -> None:
        if year not in self.record_history:
            self.record_history[year] = [0,0]
        if won: 
            self.record_history[year][0] += 1
        else: 
            self.record_history[year][1] += 1
        return 

@dataclass
class Game:
    week: int
    home: str
    away: str
    home_score: int
    away_score: int

class Simulator(object):
    def __init__(self, offinit: float = 28., definit: float = 28.) -> None:
        self.schedules = {} ##Schedules for each year. Maps year => list of games
        self.teams = {}     ##Maps team name => Team obj,
        self.team_orig = {}

        self.offinit = offinit
        self.definit = definit
        return 

    def iterate_season(self, year: int) -> Game:
        for week in self.schedules[year]:
            games_in_week = self.schedules[year][week] ##Grab that weeks games
            for game in games_in_week: ##Iterate through weekly games to update
                yield game

    def init_teams(self, fpath: str) -> None:
        unique_teams = []
        for f in fpath:
            games = pd.read_csv(f)

            #For the current models only look at P5 for more consistent results
            games = games[                     
                games.home_conference.isin(['Pac-12', 'Big Ten', 'Big 12', 'SEC', 'ACC']) &
                games.away_conference.isin(['Pac-12', 'Big Ten', 'Big 12', 'SEC', 'ACC']) 
            ].reset_index()
            unique_teams.extend(games.home_team)
            unique_teams.extend(games.away_team)
        unique_teams = sorted(list(set(unique_teams)))

        #Initialize teams with teams with the default starting offeff and defeff
        self.teams = {team:Team(team, self.offinit, self.definit) for team in unique_teams}
        self.team_orig = copy.deepcopy(self.teams)
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

    def reset_sim(self) -> None:
        self.teams = self.team_orig
        return 


#########################################
##Testing
if __name__ == "__main__":
    simulator = Simulator()
    simulator.init_teams(["../data/2019season.csv"])
    for year in range(2010, 2020):
            simulator.load_season_schedule(year, f"../data/{year}season.csv")
    for game in simulator.iterate_season(2018):
        print(game)