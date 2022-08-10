from math import frexp

import numpy as np

from dodiscover.ci import g_square_binary, g_square_discrete
from dodiscover.ci.tests import testdata

if __name__ == "__main__":

    dm = np.array([testdata.bin_data]).reshape((5000, 5))
    x = 0
    y = 1

    sets = [[], [2], [2, 3], [3, 4], [2, 3, 4]]
    for idx in range(len(sets)):
        print("x =", x, ", y =", y, ", s =", sets[idx], end="")
        _, p = g_square_binary(dm, x, y, set(sets[idx]))
        print(", p =", p, end="")
        fr_p = frexp(p)
        fr_a = frexp(testdata.bin_answer[idx])
        if round(fr_p[0] - fr_a[0], 7) == 0 and fr_p[1] == fr_a[1]:
            print(" => GOOD")
        else:
            print(" => WRONG")
            print("p =", fr_p)
            print("a =", fr_a)

    dm = np.array([testdata.dis_data]).reshape((10000, 5))
    x = 0
    y = 1

    sets = [[], [2], [2, 3], [3, 4], [2, 3, 4]]
    for idx in range(len(sets)):
        print("x =", x, ", y =", y, ", s =", sets[idx], end="")
        _, p = g_square_discrete(dm, 0, 1, set(sets[idx]), [3, 2, 3, 4, 2])
        print(", p =", p, end="")
        fr_p = frexp(p)
        fr_a = frexp(testdata.dis_answer[idx])
        if round(fr_p[0] - fr_a[0], 7) == 0 and fr_p[1] == fr_a[1]:
            print(" => GOOD")
        else:
            print(" => WRONG")
            print("p =", fr_p)
            print("a =", fr_a)

    # note that 2 and 3 are only dependent at alpha level >= 0.1
    x = 2
    y = 3
    sets = [[], [1], [0], [4], [0, 4]]
    for idx in range(len(sets)):
        print("x =", x, ", y =", y, ", s =", sets[idx], end="")
        _, p = g_square_discrete(dm, x, y, set(sets[idx]), [3, 2, 3, 4, 2])
        print(", p =", p, end="")
        fr_p = frexp(p)
        fr_a = frexp(testdata.dis_answer[idx])
        if round(fr_p[0] - fr_a[0], 7) == 0 and fr_p[1] == fr_a[1]:
            print(" => GOOD")
        else:
            print(" => WRONG")
            print("p =", fr_p)
            print("a =", fr_a)
