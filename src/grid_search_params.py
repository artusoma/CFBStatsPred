from models import KenPom, exp_squared
import itertools
import numpy as np
import nevergrad as ng
import functools

def evaluate_model(simulator: KenPom, K: float, power: int, carry_over: float):
    scores = []
    for year in range(2000, 2020):
        correct_percentage = simulator.evaluate_predicting_power(year=year, power=power, K=K)
        simulator.init_next_year(year, factor=carry_over)
        scores.append(correct_percentage)
    median = np.median(scores)
    simulator.reset_sim()
    return median

if __name__ == "__main__":
    simulator = KenPom(update_mapping=None)
    simulator.init_teams(["../data/2019season.csv"])
    for year in range(2000, 2020):
        simulator.load_season_schedule(year, f"../data/{year}season.csv")
    
    testing_model = functools.partial(evaluate_model, simulator=simulator)

    parametrization = ng.p.Instrumentation(
        K = ng.p.Log(lower=1e-4, upper=5),
        power = ng.p.Scalar(lower=2, upper=10),
        carry_over = ng.p.Scalar(lower=0, upper=1),
    )
    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=100)

    recommendation = optimizer.minimize(testing_model)

    # show the recommended keyword arguments of the function
    print(recommendation.kwargs)