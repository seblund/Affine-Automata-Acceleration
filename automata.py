from dbm import *
import time

class Location:
    def __init__(self, id):
        self.id = id
        self.transitions = []
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
        self.n_locations = locations
        self.locations = [Location(i) for i in range(locations)]

        self.locations[initial_location].extend_zone(initial_zone)
        for transition in transitions:
            self.locations[transition[0]].add_transition(transition[1:])

    def get_zones(self):
        return [location.zone for location in self.locations]

    def set_zones(self, zones):
        for l, z in zip(self.locations, zones):
            l.zone = z

    def post(self):
        post_regions = [location.post() for location in self.locations]
        post_regions = [item for sublist in post_regions for item in sublist]

        zones = self.get_zones()

        for l, dbm in post_regions:
            zones[l] = DBM.union(zones[l], dbm)

        return zones

    def REACH1(self, verbose=False):
        fixed = False
        p = None
        s = time.time()
        k = 0

        while not fixed:
            p = A.post()
            fixed = DBM.subset_all(A.post(), A.get_zones())
            A.set_zones(p)
            k += 1

        elapse = (time.time() - s)
        if verbose:
            print("--- zones ---")
            for i, zone in enumerate(A.get_zones()):
                print(f"Location {i}:")
                DBM.print_min_max(zone, ['x', 'y'])
                DBM.print_invariants(zone, ['x', 'y'])
            print("-------------")

            print(f"{elapse * 1000} ms")
            print(f"{k} iterations")
        return p



    def REACH2(self, verbose=False):
        cycles = self.find_cycles()
        pass


if __name__ == '__main__':
    T = TransformHelper(["x", "y"])

    # print(T.of("x:=3y+x+2; y:=3"))
    # exit()

    G = GuardHelper(['x', 'y'])
    ts = [
        (2, 3, G.of("x<20"), T.of("x:=x+1")),         # if x<=20 x:=x+1
        (3, 2, G.none(), T.of("y:=y+1")),              # y:=y+1
        (2, 1, G.of("y>10"), T.of("y:=x+y")),       # if y>=10 y:=x+y
        (1, 0, G.of("x<15"), T.of("x:=2x")),           # if x<=15 x:=2x
                   ]

    q = G.of("x<300, y>=200, y<=4000")
    #DBM.print(q)

    Z0 = DBM.zeros(2)

    A = Automata(4, ts, Z0, 2)

    A.REACH1(True)
