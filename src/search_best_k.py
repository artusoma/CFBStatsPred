from load_games import Season
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

if __name__ == "__main__":
    k_values = [5e-3, 1e-2, 5e-2]
    for k in k_values:
        season = Season()
        season.init_teams(["../data/2019season.csv"])

        ##Load in years
        for year in range(2010, 2020):
            season.load_season_schedule(year, f"../data/{year}season.csv")

        ##Actually iterate over, train, and evaluate
        for year in range(2010, 2019):
            season.train_elo(year=year, K=k)
            season.train_elo(year=year, K=k)
            correct_percentage = season.evaluate_predicting_power(year=(year+1), power=8)
            print("Year %d, K %f, Percentage %f" % (year+1, k, correct_percentage))

            season.save_ending_effs()
            season.init_next_year_scores(weights=np.array([1,2,3,5,20]))

    for team in ['Michigan State', 'Michigan', 'Ohio State', 'Kansas', 'Florida State', 'Penn State', 'Northwestern']:
        data = [t[0] for t in season.team_history[team]]
        plt.plot(range(len(data)), data, label=team)
    plt.legend()
    plt.show()
