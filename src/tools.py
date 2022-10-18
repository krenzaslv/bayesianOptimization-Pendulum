import math
import copy


# def rad(s):
#     r = s % (2 * math.pi)
#     if r > math.pi:
#         return -(2 * math.pi - r)
#     else:
#         return r
# return s


def clamp(t, a):
    return abs(t) / t * a if abs(t) > a else t
    # return t


def angleDiff(a, b):
    c = a - b
    return (c + math.pi) % (2 * math.pi) - math.pi
