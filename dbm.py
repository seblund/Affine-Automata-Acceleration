import numpy as np
import re

MAX_INT = np.infty
MIN_INT = -MAX_INT


# def nt0(x):
#     return np.nan_to_num(x, nan=0, posinf=np.infty, neginf=-np.infty)
#
#
def safe_product(x, y):
    if x == 0 and abs(y) == np.infty or y == 0 and abs(x) == np.infty:
        return 0
    else:
        product = x * y
        assert not np.isnan(product)
        return product


def safe_diff(x, y):
    if x == np.infty and y == np.infty:
        return np.infty
    else:
        diff = x - y
        assert not np.isnan(diff)
        return diff


to_string = np.vectorize(lambda x: 'inf' if x >= MAX_INT else '-inf' if x <= MIN_INT else f"{x:.0f}")


class Invariant:
    def __init__(self, variable, maximum=MAX_INT, minimum=MIN_INT):
        self.var = variable
        self.max = maximum
        self.min = minimum


class Guard:
    def __init__(self, n_variables, invariants):
        self.dbm = DBM.new(n_variables)
        for invariant in invariants:
            v = invariant.var + 1
            # min
            self.dbm[v, 0] = min(self.dbm[v, 0], -invariant.min)

            # max
            self.dbm[0, v] = min(self.dbm[0, v], invariant.max)


class GuardHelper:
    def __init__(self, variables):
        self.n_variables = len(variables)
        self.variables = variables

    def none(self):
        return Guard(self.n_variables, [])

    def of(self, string):
        invariants = []
        for var, comp, const in re.findall("(\\w)([<>]?=?)(-?\\d+)", string):
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

        return Guard(self.n_variables, invariants)


class TransformHelper:

    def __init__(self, variables):
        self.n_variables = len(variables)
        self.vars = variables

    def of(self, string):
        z = np.zeros(self.n_variables)
        s = np.identity(self.n_variables)
        varlst = []
        for var, expression in re.findall("(\\w):=((?:[+-]?\\w+)*)", string):
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


class DBM:
    @staticmethod
    def print(dbm):
        if dbm is None:
            print("None")
        else:
            print(to_string(DBM.convert_if_guard(dbm)))

    @staticmethod
    def zeros(n_vars):
        dbm = DBM.new(n_vars)
        dbm[:, :] = 0
        return dbm

    @staticmethod
    def new(n_vars):
        dbm = np.ones((1 + n_vars, 1 + n_vars)) * MAX_INT  # np.infty or MAX_INT if it being floats matter
        for i in range(1 + n_vars):
            dbm[i, i] = 0

        return dbm

    @staticmethod
    def subset_all(first, second):
        return np.all([DBM.subset(f, s) for f, s in zip(first, second)])

    @staticmethod
    def subset(first, second):
        if second is None:
            return first is None
        return np.all(first <= second)

    @staticmethod
    def union(first, second):
        if first is None:
            return second

        if second is None:
            return first

        # actually the convex hull of the union
        first = DBM.convert_if_guard(first)
        second = DBM.convert_if_guard(second)
        assert first.shape == second.shape
        return DBM.tighten_bounds(np.maximum(first, second))

    @staticmethod
    def intersect(first, second):
        if first is None or second is None:
            return None
        first = DBM.convert_if_guard(first)
        second = DBM.convert_if_guard(second)
        assert first.shape == second.shape
        candidate = np.minimum(first, second)
        candidate = DBM.tighten_bounds(candidate)

        if DBM.unsatisfiable(candidate):
            return None

        return candidate

    @staticmethod
    def ensure_compatible(dbm, scale, translate):
        assert dbm.shape[0] - 1 == scale.shape[0] == scale.shape[1] == translate.shape[0]
        return translate.shape[0]

    @staticmethod
    def print_min_max(dbm, variables):
        if dbm is None:
            print("None")
            return
        dbm = DBM.convert_if_guard(dbm)
        n_vars = len(variables)
        assert dbm.shape[0] - 1 == n_vars
        for i in range(n_vars):
            print(f"{to_string(-dbm[i + 1, 0])} <= {variables[i]} <= {to_string(dbm[0, i + 1])}")
            # print(f"max {variables[i]}: {to_string(dbm[0, i + 1])}")

    @staticmethod
    def print_invariants(dbm, variables, include_diagonal=False, include_min_max=False):
        if dbm is None:
            print("None")
            return
        variables = ['0'] + variables
        dbm = DBM.convert_if_guard(dbm)
        n_vars = len(variables)
        assert dbm.shape[0] == n_vars
        for i in range(n_vars):
            for j in range(n_vars):
                if not include_min_max and (i == 0 or j == 0):
                    continue
                if not include_diagonal and i != j:
                    print(f"{variables[i]} <= {variables[j]}{'+' if dbm[i, j] >= 0 else ''}"
                          f"{to_string(dbm[i, j])}")

    @staticmethod
    def transform(dbm_in, transform):
        if dbm_in is None:
            return None

        dbm_new = np.zeros_like(dbm_in)
        scale = np.asarray(transform[0])
        translate = np.asarray(transform[1])

        n_vars = DBM.ensure_compatible(dbm_in, scale, translate)

        # find new min and max values
        for var in range(n_vars):
            dbm_new[var + 1, 0] = -translate[var]
            dbm_new[0, var + 1] = translate[var]

            for n in range(n_vars):

                scaled_max = safe_product(dbm_in[0, n + 1], scale[var, n])
                scaled_min = safe_product(dbm_in[n + 1, 0], scale[var, n])
                dbm_new[var + 1, 0] += max(scaled_min, -scaled_max)  # -minimum
                dbm_new[0, var + 1] += max(-scaled_min, scaled_max)  # maximum

            assert not np.isnan(dbm_new[0, var + 1]) and not np.isnan(dbm_new[var + 1, 0])
            assert -np.infty != dbm_new[0, var + 1] and -np.infty != dbm_new[var + 1, 0]

        for ls in range(n_vars):
            for rs in range(n_vars):
                if ls == rs:
                    dbm_new[ls + 1, rs + 1] = 0
                    continue

                start = dbm_in[ls + 1, rs + 1]
                diff_ls_min = safe_diff(dbm_new[ls + 1, 0], dbm_in[ls + 1, 0])
                diff_ls_max = safe_diff(dbm_new[0, ls + 1], dbm_in[0, ls + 1])
                diff_rs_min = safe_diff(dbm_new[rs + 1, 0], dbm_in[rs + 1, 0])
                diff_rs_max = safe_diff(dbm_new[0, rs + 1], dbm_in[0, rs + 1])

                diff = safe_diff(max(diff_ls_max, -diff_ls_min), max(diff_rs_max, -diff_rs_min))

                dbm_new[ls + 1, rs + 1] = safe_diff(start, -diff)

                assert not np.isnan(dbm_new[ls + 1, rs + 1])

        return DBM.tighten_bounds(dbm_new)

    @staticmethod
    def tighten_bounds(dbm):
        n_vars = dbm.shape[0]-1
        for ls in range(n_vars):
            for rs in range(n_vars):
                if ls == rs:
                    dbm[ls + 1, rs + 1] = 0
                    continue

                # tighten other invariants based on min and maxes
                dbm[ls+1, rs+1] = min(safe_diff(dbm[0, ls + 1], -dbm[rs + 1, 0]), dbm[ls+1, rs+1])

                d = dbm[ls+1, rs+1]

                # tighten min and maxes based on other invariants
                if abs(d) != np.infty:  # we can't tighten any if infty
                    pass
                    # ls max:
                    dbm[0, ls+1] = min(dbm[0, ls+1], d + dbm[0, rs+1])
                    # rs min:
                    dbm[rs+1, 0] = min(dbm[rs+1, 0], d + dbm[ls+1, 0])  # -(-d - minx)

        return dbm  # for convenience

    @staticmethod
    def convert_if_guard(maybe_guard):
        if isinstance(maybe_guard, Guard):
            return maybe_guard.dbm
        else:
            assert isinstance(maybe_guard, np.ndarray)
            return maybe_guard

    @staticmethod
    def unsatisfiable(candidate):
        n_vars = candidate.shape[0]-1
        for i in range(n_vars):
            # minimum greater than maximum
            if candidate[0, i+1] < -candidate[i+1, 0]:
                return True

        # TODO: check diagonals
        return False
