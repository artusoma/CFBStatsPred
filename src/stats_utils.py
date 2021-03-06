import numpy as np
from simulator import Team, Game

class StatsUtils(object):

    @staticmethod
    def pythag_win_percentage(team: Team, power: float = 2, home: bool = False) -> float:
        """Pythagorean calcuation of average win percentage of a team
        """
        offeff = team.offeff
        defeff = team.defeff

        if home: 
            offeff = offeff + 2.5
            defeff = defeff - 2.5 

        pythag = offeff**power/(offeff**power+defeff**power)
        adj_pythag = pythag + (-.1+.2*pythag) 
        return adj_pythag

    @staticmethod
    def log5_win_prob(team1: Team, team2: Team, home_adv: float = 0.00) -> float:
        """Log5 calculation of probability than team1 beats team2
        """
        p1 = team1.win_p
        p2 = team2.win_p
        return (p1-p1*p2)/(p1+p2-2*p1*p2)