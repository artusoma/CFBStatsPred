from models import KenPom, exp_squared
import itertools
import numpy as np

if __name__ == "__main__":
    mapping_functions = [None, exp_squared]
    k_values = [1e-2, 5e-2, 1e-1, 5e-1, 1, 1.5, 2]
    powers = range(2, 11)
    carry_over = [.25, .5, .75, .9, 1]

    print("Total combinations: ", len(mapping_functions)*len(k_values)*len(powers)*len(carry_over))
    combinations = list(itertools.product(mapping_functions, k_values, powers, carry_over))

    scores = []
    for comb in combinations:
        f = comb[0]
        K = comb[1]
        p = comb[2]
        c = comb[3]
        
        simulator = KenPom(update_mapping=f)
        simulator.init_teams(["../data/2019season.csv"])
        for year in range(2010, 2020):
            simulator.load_season_schedule(year, f"../data/{year}season.csv")

        for year in range(2010, 2020):
            correct_percentage = simulator.evaluate_predicting_power(year=year, power=p, K=K)
            simulator.init_next_year(year, factor=c)
            scores.append(correct_percentage)
        
        print(f, K, p, c, np.mean(scores))
        scores = []