def deco(x):
    print('zobic')
    return x

@deco
def toto():
    return 0

f = deco(toto)
