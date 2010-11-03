import numpy as np 
import time

from player import players as _players

tags = {'w': 'world cup',
        'wq': 'world cup qualifier', 
        'e': 'euro', 
        'eq': 'euro qualifier',
        'c': 'confederation cup', 
        'f': 'friendly', 
        'o': 'other'}


def register_players(players):
    output = []
    for player in players:
        if isinstance(player, str):
            surname = player
            firstname = None      
        else: 
            surname = player[0]
            firstname = player[1]
        idx = [p.check(surname, firstname) for p in _players].index(True)
        output.append(_players[idx])
    return output


class Game(object): 
    
    def __init__(self, date, team, tag, location, gf, ga): 
        
        self.date = time.strptime(date, '%d %b %Y')
        self.team = str(team)
        if not tag in tags: 
            raise ValueError('Unknown tag')
        self._tag = tag 
        if location in ['home', 'away', 'neutral']: 
            self.location = location 
        else: 
            raise ValueError('Unknown location')
        self.gf = int(gf)
        self.ga = int(ga)
        self._players = []
        self._subs = []
        
    def _get_tag(self):
        return tags[self._tag] 

    tag = property(_get_tag) 

    def _get_players(self): 
        return self._players
    
    def _set_players(self, players): 
        self._players = register_players(players)

    players = property(_get_players, _set_players)

    def _get_subs(self): 
        return self._subs
    
    def _set_subs(self, players): 
        self._subs = register_players(players)

    subs = property(_get_subs, _set_subs)

    def __str__(self):
        if self.location == 'home':
            name = 'france-%s' % self.team
            score = '%d-%d' % (self.gf, self.ga)
        else: 
            name = '%s-france' % self.team
            score = '%d-%d' % (self.ga, self.gf)
        tmp = self.date
        date = str(tmp.tm_mday)+'-'+str(tmp.tm_mon)+'-'+str(tmp.tm_year)
        return '%s (%s): %s [%s]' % (name, date, score, self.tag)

    def check(self): 
        if not len(self._players) == 11: 
            return False
        return True


