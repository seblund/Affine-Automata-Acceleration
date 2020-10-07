from dbm import *
import time
from johnson import simple_cycles


class Location:
    def __init__(self, lid):
        self.id = lid
        self.transitions = []
        self.zone = None

    def reset_zone(self):
        self.zone = None

    def extend_zone(self, zone):
        self.zone = DBM.union(self.zone, zone)

    def add_transition(self, transition):
        self.transitions.append(transition)

    def post(self):
        if self.zone is None:
            return []

        posts = []
        for to, guard, transform in self.transitions:
            intersection = DBM.intersect(self.zone, guard)
            transformation = DBM.transform(intersection, transform)
            posts.append((to, transformation))

        return posts


class Automata:
    def __init__(self, locations, transitions, initial_zone, initial_location):
        self.initial_zone = initial_zone
        self.initial_location = initial_location
        self.n_locations = locations
        self.locations = [Location(i) for i in range(locations)]
        self.graph = {i: [] for i in range(locations)}
        self.locations[initial_location].extend_zone(initial_zone)
        self.n_vars = initial_zone.shape[0] - 1

        for transition in transitions:
            self.locations[transition[0]].add_transition(transition[1:])
            from_s, to_s, _, _ = transition
            self.graph[from_s].append(to_s)

        # find all cycles and order them by length
        self.cycles = list(simple_cycles(self.graph))
        self.cycles.sort(key=len)

        # find the transition (Guard and Transform) that is involved in each step of the cycle
        self.cycle_transitions = [[] for _ in range(len(self.cycles))]
        for ind, cycle in enumerate(self.cycles):
            loop = cycle + [cycle[0]]
            for i in range(len(cycle)):
                # look up the guard and transform from state i to state i+1
                gt = next(((g, t) for s_f, s_t, g, t in transitions if (s_f, s_t) == (loop[i], loop[i+1])))
                self.cycle_transitions[ind].append(gt)

        # Find the worst case bounds for completing any cycle i from any location l
        self.cycle_bounds = [[self.take_cycle(i, l, DBM.new(self.n_vars)) for l in cycle]
                             for i, cycle in enumerate(self.cycles)]

    def reset_zones(self):
        for l in self.locations:
            l.reset_zone()
        self.locations[self.initial_location].extend_zone(self.initial_zone)

    def get_zones(self):
        return [location.zone for location in self.locations]

    def set_zones(self, zones):
        for l, z in zip(self.locations, zones):
            l.zone = z

    def post(self):
        post_regions = [location.post() for location in self.locations]
        post_regions = [item for sublist in post_regions for item in sublist]

        zones = self.get_zones()  # this is a shallow copy

        for l, dbm in post_regions:
            zones[l] = DBM.union(zones[l], dbm)

        return zones

    def post_star(self):
        zones = self.get_zones()  # this is a shallow copy

        for cycle_id, cycle in enumerate(self.cycles):
            for location in cycle:
                z0 = self.locations[location].zone
                if z0 is None:
                    continue

                # take the cycle twice (handles assignments) TODO: find out if this can be avoided
                z1 = self.take_cycle(cycle_id, location, self.locations[location].zone)
                z2 = self.take_cycle(cycle_id, location, DBM.union(z0, z1))

                if z2 is None:
                    continue

                # where the value changed; accelerate the dbm to the bounds
                bounds = self.cycle_bounds[cycle_id][cycle.index(location)]
                dbm = np.where(z1 == z2, z2, bounds)
                zones[location] = DBM.union(zones[location], dbm)

        return zones

    def REACH1(self, verbose=False):
        fixed = False
        post = None
        s = time.time()
        k = 0

        while not fixed:
            k += 1
            post = self.post()

            # Determine whether we reached a fix point
            fixed = DBM.subset_all(post, self.get_zones())
            self.set_zones(post)

        elapse = (time.time() - s)
        if verbose:
            print("--- zones ---")
            for i, zone in enumerate(self.get_zones()):
                print(f"Location {i}:")
                DBM.print_min_max(zone, ['x', 'y'])
                DBM.print_invariants(zone, ['x', 'y'])
            print("-------------")

            print(f"{elapse * 1000} ms")
            print(f"{k} iterations")
        return post

    def take_cycle(self, cycle_id, start, dbm):
        cycle = self.cycles[cycle_id]
        assert start in cycle

        length = len(cycle)
        transitions = self.cycle_transitions[cycle_id]

        result = dbm
        for i in range(length):
            index = (start+i) % length
            guard, transform = transitions[index]
            result = DBM.transform(DBM.intersect(result, guard), transform)
        return result

    def REACH2(self, verbose=False):
        fixed = False
        post = None
        s = time.time()
        k = 0

        while not fixed:
            k += 1
            p_star = self.post_star()
            self.set_zones(p_star)

            # Determine whether we reached a fix point
            post = self.post()
            fixed = DBM.subset_all(post, self.get_zones())
            self.set_zones(post)

        elapse = (time.time() - s)
        if verbose:
            print("--- zones ---")
            for i, zone in enumerate(self.get_zones()):
                print(f"Location {i}:")
                DBM.print_min_max(zone, ['x', 'y'])
                DBM.print_invariants(zone, ['x', 'y'])
            print("-------------")

            print(f"{elapse * 1000} ms")
            print(f"{k} iterations")

        return post


if __name__ == '__main__':
    T = TransformHelper(["x", "y"])
    G = GuardHelper(['x', 'y'])

    ts1 = [
        # (2, 2, G.of("x>500"), T.of("x:=y")),
        (2, 3, G.of("x<500"), T.of("x:=x+1")),         # if x<=20 x:=x+1
        (3, 2, G.none(), T.of("y:=y+1")),              # y:=y+1
        (2, 1, G.of("y>10"), T.of("y:=x+y")),       # if y>=10 y:=x+y
        (1, 0, G.of("x<15"), T.of("x:=2x")),           # if x<=15 x:=2x
                   ]

    ts2 = [
        (0, 1, G.of("x<=500"), T.of("x:=x+3")),
        (1, 1, G.none(), T.of("x:=x+1")),
        (1, 2, G.of("x<=30"), T.of("y:=x+3")),
        (2, 1, G.of("x>=10; y<=100"), T.of("x:=y")),
        (2, 0, G.of("y>=10"), T.of("y:=3y+x")),
          ]

    Z0 = DBM.zeros(2)

    ts3 = [
        (0, 0, G.of("x<=y"))
    ]

    A1 = Automata(4, ts1, Z0, 2)

    # A1.REACH1(verbose=True)

    # A1.reset_zones() # reset zones before running reach again

    # A1.REACH2(verbose=True)

    A2 = Automata(3, ts2, Z0, 0)

    A1.REACH2(verbose=True)
