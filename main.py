from helpers import TransformHelper, GuardHelper
from dbm import DBM
from automata import Automata

if __name__ == '__main__':
    T = TransformHelper(["x", "y"])
    G = GuardHelper(['x', 'y'])

    ts1 = [
        # (2, 2, G.of("x>500"), T.of("x:=y")),
        (2, 3, G.of("x<500"), T.of("x:=x+1")),  # if x<=20 x:=x+1
        (3, 2, G.none(), T.of("y:=y+1")),  # y:=y+1
        (2, 1, G.of("y>10"), T.of("y:=x+y")),  # if y>=10 y:=x+y
        (1, 0, G.of("x<15"), T.of("x:=2x")),  # if x<=15 x:=2x
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