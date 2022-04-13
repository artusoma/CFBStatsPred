from simulator import Simulator
from stats_utils import StatsUtils
import numpy as np
from plotting_utils import plot_team_data
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

class KenPom(Simulator):
    def __init__(self, offinit: float = 28., definit: float = 28.) -> None:
        super().__init__(offinit, definit)
        return 

    def train_elo(self, year: int, K: float = .2) -> None:
        ##Iterate through each week in schedule
        for game in self.iterate_season(year=year):
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
            home.win_p = StatsUtils.pythag_win_percentage(home)
            away.win_p = StatsUtils.pythag_win_percentage(away)

            ##Add to histories
            home.add_history(year=year, week=game.week)
            away.add_history(year=year, week=game.week)
        return

    def evaluate_predicting_power(self, year: int, power: float = 2) -> float:
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
        return np.mean( (np.array(pythag_values) - np.array(observed_values))**2 )

##Testing
if __name__ == "__main__":
    simulator = KenPom()
    simulator.init_teams(["../data/2019season.csv"])
    for year in range(2010, 2020):
        simulator.load_season_schedule(year, f"../data/{year}season.csv")

    for year in range(2010, 2019):
        simulator.train_elo(year=year, K=.01)
        correct_percentage = simulator.evaluate_predicting_power(year=(year+1), power=2)
        print(correct_percentage)

    mse = simulator.evaluate_pythag_power(year=2018, power=2)
    print(mse)