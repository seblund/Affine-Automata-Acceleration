import random
from johnson import simple_cycles
from itertools import product


class Node:
    def __init__(self, name, writesTo: set, readsFrom: set):
        self.name = name
        self.w = writesTo
        self.r = readsFrom


def dependent_vars(node: Node, other):
    a = node.r
    b = node.w
    c = other.w
    d = node.r.union(node.w).intersection(other.w)
    return node.r.union(node.w).intersection(other.w)


def depends_on(node: Node, nodes):
    d = []
    for other in nodes:
        if other == node:
            continue
        ic = dependent_vars(node, other)
        if len(ic) > 0:
            d.append(other)
    return d


def construct_dependency_graph(nodes):
    graph = {i: [] for i in range(len(nodes))}
    for i, node in enumerate(nodes):
        for j, other in enumerate(nodes):
            if j == i:
                continue
            ic = dependent_vars(node, other)
            if len(ic) > 0:  # if the nodes depend on one another
                graph[i].append(j)
    return graph


def set_product(pools):
    result = [frozenset()]
    for pool in pools:
        result = {x.union(y) for x in result for y in pool}
    return result


def find_locking_vars(cycles, nodes):
    cycle_candidates = [set() for _ in cycles]
    for cycle_id, cycle in enumerate(cycles):
        c_l = len(cycle)
        for i in range(c_l):
            dp_vars = dependent_vars(nodes[cycle[i]], nodes[cycle[(i+1) % c_l]])
            cycle_candidates[cycle_id].add(frozenset(dp_vars))
        # cycle_candidates[cycle_id].add(frozenset({f'cycle-{cycle_id}'}))
    return cycle_candidates


def filter_strict_subsets(candidates):
    def filter_subsets(c):
        for cand in candidates:
            if cand == c:
                continue
            if cand.issubset(c):
                return False
        return True

    return filter(filter_subsets, candidates)


def find_conflicting_var_candidates(nodes, min_cycle_length=0):
    g = construct_dependency_graph(nodes)
    cycles = list(simple_cycles(g))
    cycles = list(filter(lambda c: len(c) >= min_cycle_length, cycles))

    print("Cycles:", len(cycles))

    # find the possible sets of variables one could remove to resolve conflict for each cycle
    cycle_candidates = find_locking_vars(cycles, nodes)

    # the possible products of candidates that resolve all cycles e.g. the candidate sets for conflicting variables
    var_candidates = set_product(cycle_candidates)

    print("Original candidate sets:", len(var_candidates))

    # filter out candidate sets which also have a subset as a candidate (one would never want to remove more variables)
    var_candidates = list(filter_strict_subsets(var_candidates))

    # sort for convenience
    var_candidates = sorted(var_candidates, key=len)

    return var_candidates


if __name__ == "__main__":
    variables = {chr(ord('a')+i) for i in range(26)}
    nodes = [Node(chr(ord('A')+i), set(random.sample(variables, 1)), set(random.sample(variables, 5))) for i in range(15)]

    var_candidate_sets = find_conflicting_var_candidates(nodes)

    print("Reduced Candidate sets:", len(var_candidate_sets), var_candidate_sets)

    for n in nodes:
        print(n.name, n.r, n.w)

    # for node in nodes:
    #     print(f"{node.name} depends on:")
    #     depends_on_vars = set()
    #     for dependency in depends_on(node, nodes):
    #         ic = dependent_vars(node, dependency)
    #         print("\t", dependency.name, "on", ic)
    #         depends_on_vars = depends_on_vars.union(ic)
    #     print("\tvars:", depends_on_vars)
