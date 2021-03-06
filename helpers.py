import numpy as np
import re
from dbm import DBM
from transition import PlaceholderTransform, Guard, Invariant, PlaceholderGuard


class GuardHelper:
    # TODO: represent guards as dictionaries of variables and invariants instead of invariants
    def __init__(self, variables):
        self.n_variables = len(variables)
        self.variables = variables

    def none(self):
        return Guard(self.n_variables, [])

    def of(self, string):
        invariants = []
        # TODO: allow guards of the form "x<y+10" etc. Consider allowing guards of the form "x<y+z" too
        for var, comp, const in re.findall("(\\w+)([<>]?=?)(-?\\d+)", string):
            var = self.variables.index(var)
            const = int(const)
            if comp == "<=":
                invariants.append(Invariant(var, maximum=const))
            elif comp == "<":
                invariants.append(Invariant(var, maximum=const - 1))
            elif comp == ">=":
                invariants.append(Invariant(var, minimum=const))
            elif comp == ">":
                invariants.append(Invariant(var, minimum=const + 1))
            elif comp == "=":
                invariants.append(Invariant(var, minimum=const, maximum=const))
            else:
                raise Exception(f"Unknown comparator '{comp}' of guard '{string}'")

        return Guard(self.n_variables, invariants)


class PlaceholderGuardHelper:
    def __init__(self, variables):
        self.variables = variables

    def of(self, string):
        invariants = []
        # TODO: allow guards of the form "x<y+10" etc. Consider allowing guards of the form "x<y+z" too
        for var, comp, const in re.findall("(\\w+)([<>]?=?)(-?\\d+)", string):
            assert var in self.variables
            const = int(const)
            if comp == "<=":
                invariants.append((var, None, const))
            elif comp == "<":
                invariants.append((var, None, const - 1))
            elif comp == ">=":
                invariants.append((var, const, None))
            elif comp == ">":
                invariants.append((var, const + 1, None))
            elif comp == "=":
                invariants.append((var, const, const))
            else:
                raise Exception(f"Unknown comparator '{comp}' of guard '{string}'")
        return PlaceholderGuard(invariants)

    def none(self):
        return PlaceholderGuard([])


class PlaceholderTransformHelper:
    def __init__(self, variables):
        self.variables = variables

    def of(self, string):
        transforms = []
        # [(x, [(2,x), (2,None)]), ]
        varlst = []
        for var, expression in re.findall("(\\w+):=((?:[+-]?\\w+)*)", string):
            transform = (var, [])
            varlst.append(var)
            assert var in self.variables
            for sign, coefficient, variable in re.findall("([+\\-]?)(\\d*)([A-Za-z]*)", expression)[:-1]:
                c = 1
                if coefficient != "":
                    c = int(coefficient)
                if sign == "-":
                    c *= -1
                if variable == "":
                    transform[1].append((c, None))
                else:
                    assert variable in self.variables
                    transform[1].append((c, variable))
            transforms.append(transform)
        assert len(set(varlst)) == len(varlst)
        return PlaceholderTransform(transforms)


class TransformHelper:
    # TODO: represent transforms as dictionaries of variables and transformations instead of transform matrices
    def __init__(self, variables):
        self.n_variables = len(variables)
        self.vars = variables

    def of(self, string):
        z = np.zeros(self.n_variables)
        s = np.identity(self.n_variables)
        varlst = []
        for var, expression in re.findall("(\\w+):=((?:[+-]?\\w+)*)", string):
            varlst.append(var)
            var_index = self.vars.index(var)
            s[var_index, :] = 0
            for sign, coefficient, variable in re.findall("([+\\-]?)(\\d*)([A-Za-z]*)", expression)[:-1]:
                c = 1
                if coefficient != "":
                    c = int(coefficient)
                if sign == "-":
                    c *= -1
                if variable == "":
                    assert z[self.vars.index(var)] == 0
                    z[self.vars.index(var)] = c
                else:
                    variable_index = self.vars.index(variable)
                    assert s[var_index, variable_index] == 0
                    s[var_index, variable_index] = c
        assert len(set(varlst)) == len(varlst)
        return s, z

    def add_var(self, var, other_var):
        s = np.identity(self.n_variables)
        s[var, other_var] = 1
        return s, np.zeros(self.n_variables)

    def add_n(self, var, n):
        z = np.zeros(self.n_variables)
        z[var] = n
        return np.identity(self.n_variables), z

    def plusplus(self, var):
        return self.add_n(var, 1)

    def minusminus(self, var):
        return self.add_n(var, -1)

    def scale_n(self, var, n):
        s = np.identity(self.n_variables)
        s[var, var] = n
        return s, np.zeros(self.n_variables)

    def double(self, var):
        return self.scale_n(var, 2)

    def invert(self, var):
        return self.scale_n(var, -1)
