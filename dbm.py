import numpy as np
from numpy import minimum, diagonal, newaxis
import re

MAX_INT = np.infty
MIN_INT = -MAX_INT


def safe_product(x, y):
    if x == 0 and abs(y) == np.infty or y == 0 and abs(x) == np.infty:
        return 0
    else:
        product = x * y
        assert not np.isnan(product)
        return product


def safe_diff(x, y, err=np.infty):
    if (x == np.infty and y == np.infty) or (x == -np.infty and y == -np.infty):
        return err  # The value that makes the most sense changes on the context
    else:
        diff = x - y
        assert not np.isnan(diff)
        return diff


to_string = np.vectorize(lambda x: 'inf' if x >= MAX_INT else '-inf' if x <= MIN_INT else f"{x:.0f}")


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
    def union(*dbms):
        dbms = [dbm for dbm in dbms if dbm is not None]
        n = len(dbms)
        if n == 0:
            return None
        if n == 1:
            return dbms[0]

        dbms = [DBM.convert_if_guard(dbm) for dbm in dbms]

        return DBM.tighten_bounds(np.maximum.reduce(dbms))

    @staticmethod
    def intersect(first, second):
        if first is None or second is None:
            return None
        first = DBM.convert_if_guard(first)
        second = DBM.convert_if_guard(second)
        assert first.shape == second.shape
        candidate = np.minimum(first, second)
        candidate = DBM.tighten_bounds(candidate)

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
        for j in range(n_vars):
            for i in range(n_vars):
                if not include_min_max and (i == 0 or j == 0):
                    continue
                if not include_diagonal and i != j:
                    print(f"{variables[j]} <= {variables[i]}{'+' if dbm[i, j] >= 0 else ''}"
                          f"{to_string(dbm[i, j])}")

    @staticmethod
    def transform(dbm_in, transform):
        if dbm_in is None:
            return None

        dbm_new = np.zeros_like(dbm_in)
        scale = np.asarray(transform[0])
        translate = np.asarray(transform[1])

        n_vars = DBM.ensure_compatible(dbm_in, scale, translate)

        for i in range(n_vars):
            assert scale[i, i] >= 0  # TODO: support negation by scaling

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
                    dbm_new[rs + 1, ls + 1] = 0
                    continue

                start = dbm_in[rs + 1, ls + 1]
                #diff_ls_min = safe_diff(dbm_new[ls + 1, 0], dbm_in[ls + 1, 0])
                diff_ls_max = safe_diff(dbm_new[0, ls + 1], dbm_in[0, ls + 1])
                # diff_rs_min = safe_diff(dbm_new[rs + 1, 0], dbm_in[rs + 1, 0])
                diff_rs_min = -safe_diff(dbm_new[rs + 1, 0], dbm_in[rs + 1, 0])

                # if diff_ls_max == np.infty:
                #     if scale[ls, ls] <= 1:
                #         pass  # TODO: look up the other values that were added to determine increase
                #
                # if diff_rs_min == -np.infty:
                #     if scale[rs, rs] <= 1:
                #         pass  # TODO: look up the other values that were added to determine decrease

                #diff_rs_max = safe_diff(dbm_new[0, rs + 1], dbm_in[0, rs + 1])

                # diff = safe_diff(max(diff_ls_max, -diff_ls_min), max(diff_rs_max, -diff_rs_min))
                diff = safe_diff(diff_ls_max, diff_rs_min)

                dbm_new[rs + 1, ls + 1] = safe_diff(start, -diff)  # 'safe' addition

                assert not np.isnan(dbm_new[rs + 1, ls + 1])

        return DBM.tighten_bounds(dbm_new)

    @staticmethod
    def floyd_warshall(dbm):
        # from https://gist.github.com/mosco/11178777
        # dbm[1:, 1:] = dbm[1:, 1:].transpose()
        # dbm_in = np.copy(dbm)

        dim = dbm.shape[0]

        for k in range(dim):
            dbm = minimum(dbm, dbm[newaxis, k, :] + dbm[:, k, newaxis])

        if not (diagonal(dbm) == 0.0).all():  # if the diagonal is not zero, it is unsatisfiable
            return None

        # dbm[1:, 1:] = dbm[1:, 1:].transpose()
        return dbm  # return for convenience

    @staticmethod
    def tighten_bounds(dbm):
        return DBM.floyd_warshall(dbm)  # return it for convenience

    @staticmethod
    def convert_if_guard(maybe_guard):
        from transition import Guard
        if isinstance(maybe_guard, Guard):
            return maybe_guard.dbm
        else:
            assert isinstance(maybe_guard, np.ndarray)
            return maybe_guard
