import math
import copy


def rad(x):
    # r = x % (2 * math.pi)
    # if r > math.pi:
    #     return -(2 * math.pi - r)
    # else:
    #     return r
    return x


def clamp(t, a):
    return abs(t) / t * a if abs(t) > a else t
    # return t
