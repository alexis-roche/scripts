import numpy as np

ground_floor = {
    'kitchen': 4.23*4.09,
    'entrance': 6.52*1.95 + .34*1.0, 
    'living_room': 3.66*6.5 + .4*.8 + .4*.8, 
    'laundry': 2.74*1.92 + .35*.8, 
    'shower_room': 1.15*2.12}

fireplace = .66*1.78

top_floor = {
    'master_bedroom': 3.61*3.72 - .3*.67,
    'nino_bedroom': 3.13*2.79 + .13*.97,
    'luna_bedroom': 3.11*3.92,
    'office': 1.86*3.8+(3.68-1.86)*(3.8-.32),
    'mezzanine': 1.02*3.80,
    'wc': 1.67*.90,
    'bathroom': (.74+1.71)*1.7,
    'mitigate': 2.59*2.13 + 1.72*.81 + 1.28*(2.59+.96)
}

loft = {
    'kitchen': (3.10+.88)*2.59 + .77*(2.59-.78) + 1.24*.16, 
    'shower': 1.25*.79 + .77*.70, 
    'master_bedroom': 3.36*3.05,
    'small_bedroom': 1.59*2.45 + .82*.32 + .6*.82,
    'mezzanine': 2.37*2.36
}
