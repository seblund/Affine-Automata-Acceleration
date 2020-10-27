import numpy as np
from dbm import DBM, MIN_INT, MAX_INT


class PlaceholderGuard:
    def __init__(self, invariants):
        self.invariants = invariants

    def read_set(self):
        reads = set()
        for var, _, _ in self.invariants:
            reads.add(var)
        return reads

    def initialize(self, variables):
        invariants = []
        n_variables = len(variables)

        for var, mini, maxi in self.invariants:
            invariants.append(Invariant(variables.index(var), minimum=mini, maximum=maxi))
        return Guard(n_variables, invariants)


class Invariant:
    def __init__(self, variable, maximum=MAX_INT, minimum=MIN_INT):
        self.var = variable
        self.max = maximum if maximum is not None else MAX_INT
        self.min = minimum if minimum is not None else MIN_INT


class Guard:
    def __init__(self, n_variables, invariants):
        self.dbm = DBM.new(n_variables)
        for invariant in invariants:
            v = invariant.var + 1
            # min
            self.dbm[v, 0] = min(self.dbm[v, 0], -invariant.min)

            # max
            self.dbm[0, v] = min(self.dbm[0, v], invariant.max)

        self.dbm = DBM.tighten_bounds(self.dbm)


class PlaceholderTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def write_read_sets(self):
        writes = set()
        reads = set()
        for ls_var, rs in self.transforms:
            writes.add(ls_var)
            for rs_var in rs:
                reads.add(rs_var)
        return writes, reads

    def initialize(self, variables):
        # TODO: add removal of variable list and replace by 'any'
        n_variables = len(variables)
        vars = variables

        z = np.zeros(n_variables)
        s = np.identity(n_variables)

        for var, cv in self.transforms:
            var_index = vars.index(var)
            s[var_index, :] = 0
            for coefficient, variable in cv:
                if variable is None:
                    assert z[vars.index(var)] == 0
                    z[vars.index(var)] = coefficient
                else:
                    variable_index = vars.index(variable)
                    assert s[var_index, variable_index] == 0
                    s[var_index, variable_index] = coefficient
        return s, z
