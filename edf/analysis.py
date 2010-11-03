import numpy as np 
import pylab as pl
import time

# Import games
games = []
for year in range(1994, 2010):
    print('Importing year %d' % year)
    gxx = 'g'+str(year)
    gxx_ = __import__(gxx)
    games += gxx_.__getattribute__(gxx)

# optionally, filter out friendly games
##games = filter(lambda g: not g._tag=='f', games)

# Average age
def dtime(t1, t2=None): 
    """
    t1-t2 in years
    """
    days = 365.
    if t2 == None: 
        return t1.tm_year + t1.tm_yday/days 
    else: 
        d0 = t1.tm_year - t2.tm_year
        d1 = (t1.tm_yday - t2.tm_yday)/days
        return d0+d1


def age_stat(g, stat=np.mean, use_subs=False): 
    a = [dtime(g.date, p.birthdate) for p in g.players]
    if use_subs:
        a_s = [dtime(g.date, p.birthdate) for p in g.subs]
        ages = np.asarray(a+a_s)
    else:
        ages = np.asarray(a)
    return stat(ages)

def won(games, percentage=False):
    res = np.asarray([np.sign(g.gf-g.ga) for g in games])
    won = np.sum(res>0)
    draw = np.sum(res==0)
    lost = np.sum(res<0)
    if percentage: 
        size = float(len(games))
        won = won/size
        draw = draw/size
        lost = lost/size
    return won, draw, lost

def scored(games): 
    gf = np.asarray([g.gf for g in games])
    ga = np.asarray([g.ga for g in games])
    return gf.sum(), ga.sum(), len(games)

y = np.asarray([age_stat(g) for g in games])
x = np.asarray([dtime(g.date) for g in games])


##pl.plot(x,y)

# france-bresil 98: game 52
# france-danemark 02: game 105
# france-grece 04: game 133
# france-italie 06: game 162
g0 = games[0:162]
g1 = games[0:52]
g2 = games[53:105]
g3 = games[106:162]
g4 = games[163:-1]

x0 = won(g0)
x1 = won(g1)
x2 = won(g2)
x3 = won(g3)
x4 = won(g4)

#print scored(g1)
#print scored(g2)
#print scored(g3)
#print scored(g4)

r = 10/44.

r0 = float(x0[2])/(x0[0]+x0[1]+x0[2])


print r/x1[2], r/x2[2], r/x3[2]
print r0
print r/r0

print x1
print x2
print x3
print x4 
