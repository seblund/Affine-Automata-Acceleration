from helpers import TransformHelper, GuardHelper, PlaceholderTransformHelper, PlaceholderGuardHelper
from dbm import DBM
from automata import Automata, PlaceholderAutomata

if __name__ == '__main__':
    GLOBAL_VARS = []
    LOCAL_VARS = ['x', 'y']
    ALL_VARS = GLOBAL_VARS + LOCAL_VARS
    T = PlaceholderTransformHelper(ALL_VARS)
    G = PlaceholderGuardHelper(ALL_VARS)

    ts1 = [
        # (2, 2, G.of("x>500"), T.of("x:=y")),
        (2, 3, G.of("x<500"), T.of("x:=x+1")),  # if x<=20 x:=x+1
        (2, 3, G.of("y<500"), T.of("y:=y+1")),
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

    ts3 = [
        (0, 1, G.none(), T.of("")),
        (1, 0, G.none(), T.of("")),
        (1, 1, G.none(), T.of("x:=7"))
    ]

    ts4 = [
        (0, 0, G.none(), T.of("x:=x+1")),
        (1, 1, G.none(), T.of("x:=x-1")),
        (0, 1, G.none(), T.of("")),
        (1, 0, G.none(), T.of("")),
        (1, 2, G.of("x>=500"), T.of("y:=x+y")),
        (2, 1, G.none(), T.of("x:=0")),
        (2, 3, G.of("y>=32000"), T.of(""))
    ]

    Z0 = DBM.zeros(2)

    PA1 = PlaceholderAutomata(4, ts1, 2)

    print(PA1.write_read_sets(GLOBAL_VARS))

    A1 = PA1.initialize(ALL_VARS, Z0)

    # A1.REACH1(verbose=True)

    A1.REACH2(verbose=True)

    exit()

    PA2 = PlaceholderAutomata(3, ts2, Z0, 0)

    A2 = PA2.initialize(ALL_VARS)

    # A2.REACH2(verbose=True)

    PA3 = PlaceholderAutomata(2, ts3, Z0, 0)

    A3 = PA3.initialize(ALL_VARS)

    PA4 = PlaceholderAutomata(4, ts4, Z0, 0)

    A4 = PA4.initialize(ALL_VARS)

    # A4.REACH2(True)

    test = []
    #
    # for i in range(77760000):
    #     test.append(DBM.zeros(10))
    #     if i % 1000000 == 0:
    #         print(i)
