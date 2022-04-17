import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from simulator import Team
from typing import List, Tuple, Dict
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')

def plot_team_data(teams: List[Team], min_week: int = 1, max_week: int = 14, min_year: int = 2000, max_year: int = 2019, quantity: str = 'percent'):
    data_index_map = {'percent':0, 'offeff':1, 'defeff':2}
    didx = data_index_map[quantity]

    n_data = (max_week-min_week+1)*(max_year-min_year+1)
    x_data = np.arange(n_data)
    team_ydata = pd.DataFrame(np.zeros(shape=(n_data, len(teams))))
    team_ydata.columns = [team.name for team in teams]

    ##Populate week by week data
    for team in teams:
        idx = 0
        stat_history = team.stat_history
        for year in range(min_year, max_year+1):
            for week in range(min_week, max_week+1):
                try:
                    team_ydata[team.name][idx] = stat_history[year][week][didx]
                except KeyError:
                    if idx-1 == -1:
                        team_ydata[team.name][idx] = .5
                    else:
                        team_ydata[team.name][idx] = team_ydata[team.name][idx-1]
                idx += 1
        
    for team in team_ydata:
        plt.plot(x_data, team_ydata[team], label=team, alpha=.6)

    ##Set x-labels
    year_value = [x for x in x_data if x%(max_week-min_week+1)==0]
    year_labels = [min_year+x for x in range(len(year_value))]
    plt.xticks(year_value, year_labels)

    plt.legend()
    plt.show()
    return 