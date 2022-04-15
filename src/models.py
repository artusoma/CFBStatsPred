from simulator import Simulator, Team, Game
from stats_utils import StatsUtils
import numpy as np
from plotting_utils import plot_team_data
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from typing import Callable

class KenPom(Simulator):
    def __init__(self, offinit: float = 28., definit: float = 28., update_mapping: Callable = None) -> None:
        """
        Args:
            offinit (int):
            definit (int):
            update_mapping (Callable):
        """
        super().__init__(offinit, definit)
        if update_mapping == None:
            update_mapping = self._empty_callable
        self.mapping = update_mapping
        return 

    def train_eff(self, year: int, K: float = .2) -> None:
        """Updates effective offense and defense without recording model
        performance. 
        """
        ##Iterate through each week in schedule
        for game in self.iterate_season(year=year):
            self.update_eff(game=game, K=K)
        return

    def update_eff(self, game: Game, year: int, K: float = .2):
        """From a single game update the teams offeff and defeff. 
        """
        
        home = self.teams[game.home] ##Grab team objects from team names
        away = self.teams[game.away]

        ##Adjust home offeff
        expected_home_offense = (home.offeff + away.defeff)/2
        home.offeff = home.offeff + K*(self.mapping(game.home_score-expected_home_offense))

        ##Adjust away offeff
        expected_away_offense = (away.offeff + home.defeff)/2
        away.offeff = away.offeff + K*(self.mapping(game.away_score-expected_away_offense))
        home.defeff = home.defeff + K*(self.mapping(game.away_score-expected_away_offense)) ##Adjust home defeff
        away.defeff = away.defeff + K*(self.mapping(game.home_score-expected_home_offense)) ##Adjust away defeff

        ##Now, update winning percentages
        home.win_p = StatsUtils.pythag_win_percentage(home)
        away.win_p = StatsUtils.pythag_win_percentage(away)

        ##Add to histories
        home.add_history(year=year, week=game.week)
        away.add_history(year=year, week=game.week)
        return 

    def evaluate_predicting_power(self, year: int, power: float = 2., K: float = 2.) -> float:
        """Iterate a season and keep track of how the model predicts vs the actual outcome of 
        the game. 
        """
        ncorrect = 0
        total = 0
        for team_name in self.teams:
            team_obj = self.teams[team_name]
            team_obj.win_p = StatsUtils.pythag_win_percentage(team_obj, power=power)

        for game in self.iterate_season(year=year):
            team1 = self.teams[game.home]
            team2 = self.teams[game.away]
            #print(f"{game.home} vs {game.away} :: {game.home_score}-{game.away_score} :: Predicted {log5_win_prob(team1, team2)}")

            if (game.home_score > game.away_score) & (StatsUtils.log5_win_prob(team1, team2) > .5):
                ncorrect += 1
            if (game.home_score < game.away_score) & (StatsUtils.log5_win_prob(team1, team2) < .5):
                ncorrect += 1
            total += 1

            self.update_eff(game=game, year=year, K=K)

        return ncorrect/total

    def evaluate_pythag_power(self, year: int, power: float = 2) -> float:
        ##Compute updated winning percentage with given power
        for team_name in self.teams:
            team_obj = self.teams[team_name]
            team_obj.win_p = StatsUtils.pythag_win_percentage(team_obj, power=power)

        ##Iterate through games and see which teams won and lost
        for game in self.iterate_season(year=year):
            if (game.home_score > game.away_score):
                self.teams[game.home].update_wl(year=year, won=True)
                self.teams[game.away].update_wl(year=year, won=False)
            else:
                self.teams[game.home].update_wl(year=year, won=False)
                self.teams[game.away].update_wl(year=year, won=True)

        ##Compare these values to that given by pythag
        pythag_values = []
        observed_values = []
        for team_name in self.teams:
            team_obj = self.teams[team_name]
            observed_values.append(team_obj.get_record(year=year))
            pythag_values.append(team_obj.win_p)
        return np.mean((np.array(pythag_values) - np.array(observed_values))**2)

    def init_next_year(self, prev_year: int, factor: float = .5):
        for team in self.teams:
            try:
                last_year_stats = self.teams[team].stat_history[prev_year]
            except KeyError:
                continue
            final_week = max(week for week, stats in last_year_stats.items())
            prev_off, prev_def = last_year_stats[final_week][1:3]
            new_off = prev_off*factor + self.offinit*(1-factor)
            new_def = prev_def*factor + self.definit*(1-factor)
            self.teams[team].offeff = new_off
            self.teams[team].defeff = new_def
        return

    def _empty_callable(self, value: float) -> float:
        return value

def exp_squared(value: float) -> float:
    if value >= 0:
        return -21*np.exp(-0.005*value**2) + 21
    if value < 0:
        return 21*np.exp(-0.005*value**2) - 21

##Testing
if __name__ == "__main__":
    simulator = KenPom(update_mapping=None)
    simulator.init_teams(["../data/2019season.csv"])
    for year in range(2010, 2020):
        simulator.load_season_schedule(year, f"../data/{year}season.csv")

    for year in range(2010, 2020):
        correct_percentage = simulator.evaluate_predicting_power(year=year, power=3, K=.1)
        simulator.init_next_year(year, factor=.5)
        print(correct_percentage)