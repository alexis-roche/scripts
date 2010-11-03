import sympy 
from sympy import Symbol
import numpy as np



class LinearModel:
    
    def __init__(self, symbols, hrf): 
        self.symbols = symbols + [Symbol('drift')]
        self.hrf = hrf
        self.regressors = {}

    def set_condition(self, symbol, onsets, amplitudes=None, durations=None): 
        if self.symbols.count(symbol): 
            for f in self.hrf:
                self.regressors[symbol] = 'ok'
                # events(onsets, amplitudes=amplitudes, g=symbol, f=f)
                # or blocks(...)

    def set_drift(self, order=3, expression=None): 
        d = []
        if isinstance(expression, sympy.function.FunctionClass): 
            monom = expression(Symbol('time'))
        else: 
            monom = Symbol('time')

        for i in range(order): 
            d.append(monom**i)
        self.regressors[Symbol('drift')] = d

    def set_regressor(self): 
        return

    def design_matrix(self, timestamps): 
        return 

    def formula(self): 
        return 

                    
